// metal/src/engine/msl/mtp_chain.metal - the two steps that keep a whole MTP draft round on the GPU (one command
// buffer for every draft step; compiled after mac-kern's quant.metal and units.metal):
//
//   mtp_pick     draft j: the pick's token id (through the draft vocabulary) and its probability, the softmax of
//                the draft head's logits at the pick (the host computed it after every step before)
//   mtp_embed_F  the next step's input: the picked token's embedding row dequantized from the model's token_embd
//                (format F) - the host dequantized it before
//
// mtp_embed reads a row with mac-kern's unit decoders (units.metal Dec<F>): a unit's dot with x is
//   d1 * sum_{j < UNIT/2} q[j] xs[j] + d2 * sum_{j >= UNIT/2} q[j] xs[j] - (m1 xsum0 + m2 xsum1)
// with xs the prepared x (raw x times the kind's 2^-k) and xsum the raw sums of 16, so weight j of the unit is
//   w = d * (q[j] * 2^-k) - m        (d = d1 or d2 by half, m = m1 for the first 16 x values, m2 for the rest;
//                                     a 16-wide unit has one sum: m1)
// and its position in the row is what X<F>::load reads when x holds the positions 0 .. n-1.

struct PickArgs {
    uint nv;        // the draft head's rows (the draft vocabulary, or the whole vocabulary)
    uint j;         // the draft step
    uint use_map;   // 1: map the head's row through the draft vocabulary
};

// one threadgroup of 1024
kernel void mtp_pick(device const float* lg [[buffer(0)]], device const int* pick [[buffer(1)]],
                     device const int* vmap [[buffer(2)]], device int* dout [[buffer(3)]], device float* dprob [[buffer(4)]],
                     constant PickArgs& a [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    const int di = pick[0];
    const float mx = lg[di];
    float s = 0.0f;
    for (uint i = tid; i < a.nv; i += 1024) s += exp(lg[i] - mx);
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (int k = 0; k < 32; ++k) t += part[k];
        dout[a.j] = a.use_map ? vmap[di] : di;
        dprob[a.j] = 1.0f / t;
    }
}

struct EmbArgs {
    uint n;           // the row's width
    uint row_bytes;
    uint j;           // dout[j] is the token
};

template <int F> static inline float kind_scale(int pos) {
    const int k = XKind<F>::K;
    if (k == XK_Q2) {
        const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
        return q2s[pos & 7];
    }
    if (k == XK_P4) {
        const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
        return p4s[pos & 3];
    }
    if (k == XK_P2) return (pos & 1) ? 0.00390625f : 1.0f;
    return 1.0f;
}

// one threadgroup of 256; posv = the floats 0 .. n-1
template <int F>
kernel void mtp_embed(device const uchar* W [[buffer(0)]], device const int* dout [[buffer(1)]],
                      device const float* posv [[buffer(2)]], device float* y [[buffer(3)]], constant EmbArgs& a [[buffer(4)]],
                      ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, H = UNIT / 2;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar* row = W + (ulong) dout[a.j] * a.row_bytes;
    const int U = int(a.n) / UNIT;
    for (int u = tid; u < U; u += 256) {
        float q[UNIT], ps[UNIT], xsum[2];
        float4 sc;
        Dec<F>::run(row, u, q, sc, tab);
        X<F>::load(posv, posv, u, ps, xsum);
        for (int j = 0; j < UNIT; ++j) {
            const int pos = int(ps[j]);
            const float d = j < H ? sc.x : sc.y;
            const float m = (UNIT == 16 || j < 16) ? sc.z : sc.w;
            y[pos] = d * (q[j] * kind_scale<F>(pos)) - m;
        }
    }
}

// host names by the format's GGUF type number (mtp_embed_11 = Q3_K): EMB expands F before EMB_ stringifies it
#define EMB_(F) template [[host_name("mtp_embed_" #F)]] kernel decltype(mtp_embed<F>) mtp_embed<F>;
#define EMB(F) EMB_(F)
EMB(T_BF16) EMB(T_Q8_0) EMB(T_Q4_0) EMB(T_Q5_0) EMB(T_IQ4_NL) EMB(T_IQ4_XS) EMB(T_Q2_0) EMB(T_IQ3_S) EMB(T_Q3_K)
EMB(T_Q4_K) EMB(T_Q5_K) EMB(T_Q6_K) EMB(T_IQ3_XXS)
