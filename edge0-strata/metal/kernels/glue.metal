// metal/kernels/glue.metal - small elementwise steps between the big kernels (after quant/units/moe.metal).
//
//   swiglu_prep    h = silu(g) * u for the shared expert, written as prepared activations (xprep layout)
//   moe_combine_sh bo[t] = sum_k ye[entry(t, k)] + sigmoid(sg[t]) * ysh[t]   (k in order)

struct GlueArgs {
    uint n;        // vector width
    uint n_tok;
    uint n_embd;
    uint topk;
};

// grid (n / 16, n_tok), 16 threads: one group of 16 values
kernel void swiglu_prep(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                        device float* xp [[buffer(2)]], constant GlueArgs& a [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort j [[thread_index_in_threadgroup]]) {
    threadgroup float v[16];
    const uint t = tg.y, d = tg.x * 16 + j, plane = a.n_tok * a.n;
    const float gv = g[t * a.n + d];
    const float h = gv / (1.0f + exp(-gv)) * u[t * a.n + d];
    const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
    const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
    device float* raw = xp + t * a.n + d;
    raw[0] = h;
    raw[plane] = h * q2s[j & 7];
    raw[2 * plane] = h * p4s[j & 3];
    raw[3 * plane] = (j & 1) ? h * 0.00390625f : h;
    v[j] = h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0) {
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += v[k]; s1 += v[k + 1]; }
        xp[4 * plane + t * (a.n / 16) + tg.x] = s0 + s1;
    }
}

kernel void moe_combine_sh(device const float* ye [[buffer(0)]], device const int* plan [[buffer(1)]],
                           device const float* ysh [[buffer(2)]], device const float* sg [[buffer(3)]],
                           device float* bo [[buffer(4)]], constant GlueArgs& a [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.n_embd, k = gid % a.n_embd;
    if (t >= a.n_tok) return;
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += ye[plan[P_TOKENT + t * a.topk + j] * int(a.n_embd) + k];
    const float gs = 1.0f / (1.0f + exp(-sg[t]));
    bo[gid] = fma(gs, ysh[gid], s);
}

// slc_touch: reads n16 16-byte words (a weight range the next kernels will need) so they sit in the SLC; the xor is
// written only if it equals a value it never takes (keeps the loads alive).  grid (ceil(n16 / 1024)), 256 threads,
// 4 words a thread.
kernel void slc_touch(device const uint4* src [[buffer(0)]], device uint* sink [[buffer(1)]], constant uint& n16 [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint s = 0;
    for (uint k = 0; k < 4; ++k) {
        const uint j = (gid >> 8) * 1024 + k * 256 + (gid & 255);
        if (j < n16) {
            const uint4 v = src[j];
            s ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (s == 0x9e3779b9u) sink[0] = s;
}
