// metal/kernels/sample.metal - greedy sampling over the LM head's logits: the largest logit, ties to the lowest
// token id (NaN never wins).  Two stages: argmax_part (grid (parts, n_tok), 256 threads) writes one (value, id)
// pair per threadgroup; argmax_final (grid (n_tok), 256 threads) reduces them in order.

struct ArgArgs {
    uint n_vocab;
    uint parts;
};

static inline void better(thread float& bv, thread int& bi, float v, int i) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

static inline void tg_argmax(thread float& bv, thread int& bi, threadgroup float* sv, threadgroup int* si, ushort sg,
                             ushort lane) {
    for (ushort m = 16; m > 0; m >>= 1) {
        const float ov = simd_shuffle_xor(bv, m);
        const int oi = simd_shuffle_xor(bi, m);
        better(bv, bi, ov, oi);
    }
    if (lane == 0) { sv[sg] = bv; si[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = -INFINITY;
    bi = 0x7fffffff;
    for (int k = 0; k < 8; ++k) better(bv, bi, sv[k], si[k]);
}

kernel void argmax_part(device const float* logits [[buffer(0)]], device float* pv [[buffer(1)]],
                        device int* pi [[buffer(2)]], constant ArgArgs& a [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup int si[8];
    const uint t = tg.y, per = (a.n_vocab + a.parts - 1) / a.parts;
    const uint b0 = tg.x * per, b1 = min(b0 + per, a.n_vocab);
    device const float* l = logits + t * a.n_vocab;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint i = b0 + tid; i < b1; i += 256) {
        const float v = l[i];
        if (!isnan(v)) better(bv, bi, v, int(i));
    }
    tg_argmax(bv, bi, sv, si, sg, lane);
    if (tid == 0) { pv[t * a.parts + tg.x] = bv; pi[t * a.parts + tg.x] = bi; }
}

kernel void argmax_final(device const float* pv [[buffer(0)]], device const int* pi [[buffer(1)]],
                         device int* out [[buffer(2)]], constant ArgArgs& a [[buffer(3)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[8];
    threadgroup int si[8];
    const uint t = tg.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint k = tid; k < a.parts; k += 256) better(bv, bi, pv[t * a.parts + k], pi[t * a.parts + k]);
    tg_argmax(bv, bi, sv, si, sg, lane);
    if (tid == 0) out[t] = bi;
}
