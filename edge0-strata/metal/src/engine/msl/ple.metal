// metal/src/engine/msl/ple.metal - the PLE block (include/strata/kernels/ple.hpp) after its two projections, for a
// window of T rows, on the GPU (the CUDA engine's ple.cu math; its block sums are double there, float here):
//
//   key   = grouped_norm(pkey, norm_key)          per stream c of 4: x * rsqrt(mean(x^2) + eps) * w
//   query = grouped_norm(R, norm_query)
//   gate[c] = sigmoid(sign(s) sqrt(max(|s|, 1e-6))),  s = key[c] . query[c] / sqrt(n_embd)
//   gated = value[d] * gate[c];   normalized = grouped_norm(gated, norm_conv)
//   conv[c] = silu(sum_k kW[k + 4c] * row(t - (3 - k) * 3)[c])   rows: the history, then the window's own
//   R += gated + conv
//
// History: hist[row + 9 * channel] (row-fastest, oldest first, the committed rows); the window's normalized rows
// are written to norm_w[t][channel] (the host appends the kept ones at commit).

#include <metal_stdlib>
using namespace metal;

struct PleArgs {
    uint n_embd;   // 2560
    uint hc;       // 4
    float eps;
};

static inline float tg_sum(float v, threadgroup float* part, ushort sg, ushort lane) {
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (int i = 0; i < 8; ++i) t += part[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// grid (hc, T), 256 threads: y[t][c] = grouped_norm(x[t][c]) with w[c]
kernel void ple_gnorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
                      constant PleArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    device const float* xc = x + (t * a.hc + c) * n;
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) ss += xc[d] * xc[d];
    const float mean = tg_sum(ss, part, sg, lane) / float(n);
    const float scale = 1.0f / sqrt(mean + a.eps);
    device float* yc = y + (t * a.hc + c) * n;
    for (uint d = tid; d < n; d += 256) yc[d] = xc[d] * scale * w[c * n + d];
}

// grid (hc, T), 256 threads: the gate, then gated = value * gate and normalized = grouped_norm(gated, norm_conv)
kernel void ple_gate_norm(device const float* key [[buffer(0)]], device const float* query [[buffer(1)]],
                          device const float* value [[buffer(2)]], device const float* w_conv [[buffer(3)]],
                          device float* gated [[buffer(4)]], device float* norm_w [[buffer(5)]],
                          constant PleArgs& a [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd, row = (t * a.hc + c) * n;
    float acc = 0.0f;
    for (uint d = tid; d < n; d += 256) acc += key[row + d] * query[row + d];
    const float s = tg_sum(acc, part, sg, lane) * (1.0f / sqrt(float(n)));
    const float mag = sqrt(max(fabs(s), 1e-6f));
    const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
    const float gate = 1.0f / (1.0f + exp(-(sgn * mag)));
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        const float g = value[t * n + d] * gate;
        gated[row + d] = g;
        ss += g * g;
    }
    const float mean = tg_sum(ss, part, sg, lane) / float(n);
    const float scale = 1.0f / sqrt(mean + a.eps);
    for (uint d = tid; d < n; d += 256) norm_w[row + d] = gated[row + d] * scale * w_conv[c * n + d];
}

// grid (hc * n_embd / 256, T), 256 threads: the conv over [history | the window's rows], SiLU, and the residual
// PLE_ROWS (function constant 32): the rows are different sequences (batch slots): row t convolves its slot's history only (slot
// stride rinfo[7] floats, row t at rinfo[8 + 4 t] = {slot, ...}); the three taps before the row itself always read the history.
constant bool PLE_ROWS_C [[function_constant(32)]];
constant bool ple_rows = is_function_constant_defined(PLE_ROWS_C) && PLE_ROWS_C;

kernel void ple_conv_add(device const float* hist [[buffer(0)]], device const float* norm_w [[buffer(1)]],
                         device const half* kw [[buffer(2)]], device const float* gated [[buffer(3)]],
                         device float* R [[buffer(4)]], constant PleArgs& a [[buffer(5)]],
                         device const uint* rinfo [[buffer(6), function_constant(ple_rows)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint ch = tg.x * 256 + tid, t = tg.y, hcd = a.hc * a.n_embd;
    if (ch >= hcd) return;
    if (ple_rows) hist += ulong(rinfo[8 + 4 * t]) * rinfo[7];
    float acc = 0.0f;
    for (int k = 0; k < 4; ++k) {
        float v;
        if (ple_rows) {
            v = k == 3 ? norm_w[t * hcd + ch] : hist[uint(9 - (3 - k) * 3) + 9 * ch];
        } else {
            const int p = int(t) - (3 - k) * 3;   // the row this tap reads: < 0 = the committed history
            v = p >= 0 ? norm_w[uint(p) * hcd + ch] : hist[uint(9 + p) + 9 * ch];
        }
        acc += float(kw[k + 4 * ch]) * v;
    }
    const float conv = acc / (1.0f + exp(-acc));
    device float* r = R + t * hcd;
    r[ch] = r[ch] + gated[t * hcd + ch] + conv;
}
