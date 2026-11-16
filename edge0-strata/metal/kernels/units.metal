// metal/kernels/units.metal - the dot of one unit with the window's tokens (after quant.metal).
//
// X<F>::load(xrow, u, xs, xsum) prepares unit u's x slice of one token (xrow = the token's row start), once per unit
// and shared by every weight row the lane reads: the UNIT x values in the order the decoder produces weights, and
// up to two partial sums (for the formats with an offset or a min).  Where the codes are read with the mask trick
// (an AND keeps a field in place, so the code arrives multiplied by 2^k), the x values are pre-scaled by 2^-k here:
// exact in float.

// the threadgroup-memory form (x staged once per threadgroup; the same values)
static inline void load16(threadgroup const float* p, thread float* xs) {
    threadgroup const float4* v = (threadgroup const float4*)p;
    for (int j = 0; j < 4; ++j) {
        const float4 a = v[j];
        xs[4 * j] = a.x;
        xs[4 * j + 1] = a.y;
        xs[4 * j + 2] = a.z;
        xs[4 * j + 3] = a.w;
    }
}
static inline void load16(device const float* p, thread float* xs) {
    device const float4* v = (device const float4*)p;
    for (int j = 0; j < 4; ++j) {
        const float4 a = v[j];
        xs[4 * j] = a.x;
        xs[4 * j + 1] = a.y;
        xs[4 * j + 2] = a.z;
        xs[4 * j + 3] = a.w;
    }
}

// ---- prepared x (xprep): XP holds, per token row of n floats, the kinds
//   XK_RAW  x                                  (every format without the mask trick)
//   XK_Q2   x * 4^-(i % 8)                     (Q2_0: 2-bit fields in place in a 16-bit word)
//   XK_P4   x * 2^-(8 (i % 4))                 (Q4_K, Q5_K: byte fields in place in a 32-bit word)
//   XK_P2   x * 2^-(8 (i % 2))                 (Q6_K, Q3_K: byte fields in place in a 16-bit word)
// and S16 = the sum of every 16 consecutive raw x (the formats with an offset or a min).
#define XK_RAW 0
#define XK_Q2 1
#define XK_P4 2
#define XK_P2 3
template <int F> struct XKind { enum { K = XK_RAW }; };
template <> struct XKind<T_Q2_0> { enum { K = XK_Q2 }; };
template <> struct XKind<T_Q4_K> { enum { K = XK_P4 }; };
template <> struct XKind<T_Q5_K> { enum { K = XK_P4 }; };
template <> struct XKind<T_Q6_K> { enum { K = XK_P2 }; };
template <> struct XKind<T_Q3_K> { enum { K = XK_P2 }; };

// X<F>::load(xp, s16, u, xs, xsum): unit u's prepared x (xp = the token's row of the format's kind, s16 = its sums)
template <int F> struct X {
    template <typename P> static inline void load(P xp, P, int u, thread float* xs, thread float* xsum) {
        constexpr int UNIT = fmt<F>::UNIT;
        for (int k = 0; k < UNIT; k += 16) load16(xp + u * UNIT + k, xs + k);
        xsum[0] = xsum[1] = 0.0f;
    }
};
template <> struct X<T_Q2_0> {
    template <typename P> static inline void load(P xp, P s16, int u, thread float* xs, thread float* xsum) {
        load16(xp + 16 * u, xs);
        xsum[0] = s16[u];
        xsum[1] = 0.0f;
    }
};
// Q4_K / Q5_K: unit (block b, chunk c, half m): run0 at 256b + 64c + 16m, run1 = run0 + 32
template <> struct X<T_Q4_K> {
    template <typename P> static inline void load(P xp, P s16, int u, thread float* xs, thread float* xsum) {
        const int p0 = 256 * (u >> 3) + 64 * ((u >> 1) & 3) + 16 * (u & 1);
        load16(xp + p0, xs);
        load16(xp + p0 + 32, xs + 16);
        xsum[0] = s16[p0 >> 4];
        xsum[1] = s16[(p0 >> 4) + 2];
    }
};
template <> struct X<T_Q5_K> {
    template <typename P> static inline void load(P xp, P s16, int u, thread float* xs, thread float* xsum) {
        X<T_Q4_K>::load(xp, s16, u, xs, xsum);
    }
};
// Q6_K: unit (block b, half n, m, p): run0 at 256b + 128n + 32p + 16m, run1 = run0 + 64
template <> struct X<T_Q6_K> {
    template <typename P> static inline void load(P xp, P s16, int u, thread float* xs, thread float* xsum) {
        const int p0 = 256 * (u >> 3) + 128 * ((u >> 2) & 1) + 32 * (u & 1) + 16 * ((u >> 1) & 1);
        load16(xp + p0, xs);
        load16(xp + p0 + 64, xs + 16);
        xsum[0] = s16[p0 >> 4];
        xsum[1] = s16[(p0 >> 4) + 4];
    }
};
// Q5_1: 32 contiguous raw x, the two per-16 sums formed from them (the GEMVs that read Q5_1 - the routed down - get
// h without prepared sums)
template <> struct X<T_Q5_1> {
    template <typename P> static inline void load(P xp, P, int u, thread float* xs, thread float* xsum) {
        load16(xp + 32 * u, xs);
        load16(xp + 32 * u + 16, xs + 16);
        float a = 0.0f, b = 0.0f;
        for (int j = 0; j < 16; ++j) { a += xs[j]; b += xs[16 + j]; }
        xsum[0] = a;
        xsum[1] = b;
    }
};
// Q3_K: 32 contiguous, sums per 16
template <> struct X<T_Q3_K> {
    template <typename P> static inline void load(P xp, P s16, int u, thread float* xs, thread float* xsum) {
        load16(xp + 32 * u, xs);
        load16(xp + 32 * u + 16, xs + 16);
        xsum[0] = s16[2 * u];
        xsum[1] = s16[2 * u + 1];
    }
};

// xprep: one thread per 16 consecutive values of one token row; XP = [4 kinds][n_tok][n] then S16 [n_tok][n / 16]
kernel void xprep(device const float* x [[buffer(0)]], device float* xp [[buffer(1)]], constant uint& n [[buffer(2)]],
                  constant uint& n_tok [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint per = n / 16;
    if (gid >= per * n_tok) return;
    const uint t = gid / per, i = gid % per;
    device const float* src = x + t * n + 16 * i;
    const uint plane = n_tok * n;
    device float* raw = xp + t * n + 16 * i;
    float v[16];
    load16(src, v);
    float s0 = 0, s1 = 0;
    for (int j = 0; j < 16; j += 2) { s0 += v[j]; s1 += v[j + 1]; }
    const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
    const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
    for (int j = 0; j < 16; ++j) {
        raw[j] = v[j];
        raw[plane + j] = v[j] * q2s[j & 7];
        raw[2 * plane + j] = v[j] * p4s[j & 3];
        raw[3 * plane + j] = (j & 1) ? v[j] * 0.00390625f : v[j];
    }
    xp[4 * plane + t * per + i] = s0 + s1;
}

// ---- decoders: Dec<F>::run(row, u, q, sc) decodes unit u of a row into q[UNIT] and sc = (d1, d2, m1, m2), so that
// for any token: dot = d1 * sum_{j < UNIT/2} q[j] xs[j] + d2 * sum_{j >= UNIT/2} q[j] xs[j] - (m1 xsum0 + m2 xsum1)
// with xs, xsum from X<F>::load.  A unit's weights are decoded once per row and applied to every token.
template <int F> struct Dec {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint* tab) {
        constexpr int UNIT = fmt<F>::UNIT, UPB = fmt<F>::QK / UNIT;
        const float s = deq<F>(row + (u / UPB) * fmt<F>::BB, u % UPB, q, tab);
        sc = float4(s, s, 0.0f, 0.0f);
    }
};

template <> struct Dec<T_Q2_0> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 2) * 18;
        device const ushort* p = (device const ushort*)(b + 2 + 4 * (u & 3));
        const ushort a = p[0], c = p[1];
        for (int j = 0; j < 8; ++j) {
            q[j] = float(a & ushort(3u << (2 * j)));
            q[8 + j] = float(c & ushort(3u << (2 * j)));
        }
        const float d = rdh(b);
        sc = float4(d, d, d, 0.0f);
    }
};

// Q2_0 with raw x: q = c - 1 (shift, and, convert, subtract), one scale
template <> struct Dec<T_Q2_0R> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 2) * 18;
        device const ushort* p = (device const ushort*)(b + 2 + 4 * (u & 3));
        const ushort a = p[0], c = p[1];
        for (int j = 0; j < 8; ++j) {
            q[j] = float((a >> (2 * j)) & 3) - 1.0f;
            q[8 + j] = float((c >> (2 * j)) & 3) - 1.0f;
        }
        const float d = rdh(b);
        sc = float4(d, d, 0.0f, 0.0f);
    }
};

static inline void scale_min_k4(int j, device const uchar* q, thread float& d, thread float& m) {
    if (j < 4) {
        d = float(q[j] & 63);
        m = float(q[j + 4] & 63);
    } else {
        d = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
static inline void bytes4(uint w, thread float* q) {
    q[0] = float(w & 0xFFu);
    q[1] = float(w & 0xFF00u);
    q[2] = float(w & 0xFF0000u);
    q[3] = float(w & 0xFF000000u);
}
static inline void bytes2(ushort w, thread float* q) {
    q[0] = float(w & 0xFF);
    q[1] = float(w & 0xFF00);
}

template <> struct Dec<T_Q5_1> {   // w = d q + m:  sum = d (q . x) + m (sum x)
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint* tab) {
        device const uchar* b = row + u * 24;
        const float d = deq<T_Q5_1>(b, 0, q, tab), m = rdh(b + 2);
        sc = float4(d, d, -m, -m);
    }
};

template <> struct Dec<T_Q4_K> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 144;
        const int c = (u >> 1) & 3, m = u & 1;
        const float d = rdh(b), dm = rdh(b + 2);
        float s1, m1, s2, m2;
        scale_min_k4(2 * c, b + 4, s1, m1);
        scale_min_k4(2 * c + 1, b + 4, s2, m2);
        device const uint* qp = (device const uint*)(b + 16 + 32 * c + 16 * m);
        for (int w = 0; w < 4; ++w) {
            const uint v = qp[w];
            bytes4(v & 0x0F0F0F0Fu, q + 4 * w);
            bytes4((v >> 4) & 0x0F0F0F0Fu, q + 16 + 4 * w);
        }
        sc = float4(d * s1, d * s2, dm * m1, dm * m2);
    }
};

template <> struct Dec<T_Q5_K> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 176;
        const int c = (u >> 1) & 3, m = u & 1;
        const float d = rdh(b), dm = rdh(b + 2);
        float s1, m1, s2, m2;
        scale_min_k4(2 * c, b + 4, s1, m1);
        scale_min_k4(2 * c + 1, b + 4, s2, m2);
        device const uint* qp = (device const uint*)(b + 48 + 32 * c + 16 * m);
        device const uint* hp = (device const uint*)(b + 16 + 16 * m);
        for (int w = 0; w < 4; ++w) {
            const uint v = qp[w], h = hp[w];
            bytes4((v & 0x0F0F0F0Fu) | (((h >> (2 * c)) & 0x01010101u) << 4), q + 4 * w);
            bytes4(((v >> 4) & 0x0F0F0F0Fu) | (((h >> (2 * c + 1)) & 0x01010101u) << 4), q + 16 + 4 * w);
        }
        sc = float4(d * s1, d * s2, dm * m1, dm * m2);
    }
};

template <> struct Dec<T_Q6_K> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 210;
        const int n = (u >> 2) & 1, m = (u >> 1) & 1, p = u & 1;
        device const ushort* ql = (device const ushort*)(b + 64 * n + 32 * p + 16 * m);
        device const ushort* qh = (device const ushort*)(b + 128 + 32 * n + 16 * m);
        device const char* scs = (device const char*)(b + 192);
        const float d = rdh(b + 208);
        for (int w = 0; w < 8; ++w) {
            const ushort l = ql[w], h = qh[w];
            bytes2((l & 0x0F0F) | (((h >> (2 * p)) & 0x0303) << 4), q + 2 * w);
            bytes2(((l >> 4) & 0x0F0F) | (((h >> (2 * p + 4)) & 0x0303) << 4), q + 16 + 2 * w);
        }
        const float d1 = d * float(scs[8 * n + m + 2 * p]), d2 = d * float(scs[8 * n + m + 2 * p + 4]);
        sc = float4(d1, d2, 32.0f * d1, 32.0f * d2);
    }
};

template <> struct Dec<T_Q3_K> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 110;
        const int n = (u >> 2) & 1, j = u & 3;
        // qs and hmask as packed (2-byte aligned) vectors: blocks are 110 bytes, so odd blocks are only 2-aligned
        device const packed_ushort4* qv = (device const packed_ushort4*)(b + 32 + 32 * n);
        device const packed_ushort4* hv = (device const packed_ushort4*)b;
        ushort qq[16], hm[16];
        for (int v = 0; v < 4; ++v) {
            const ushort4 a4 = ushort4(qv[v]), h4 = ushort4(hv[v]);
            qq[4 * v] = a4.x; qq[4 * v + 1] = a4.y; qq[4 * v + 2] = a4.z; qq[4 * v + 3] = a4.w;
            hm[4 * v] = h4.x; hm[4 * v + 1] = h4.y; hm[4 * v + 2] = h4.z; hm[4 * v + 3] = h4.w;
        }
        device const uchar* s = b + 96;
        const int i1 = 8 * n + 2 * j, i2 = i1 + 1;
        const int l1 = (i1 < 8 ? s[i1] & 0xF : s[i1 - 8] >> 4) | (((s[8 + (i1 & 3)] >> (2 * (i1 >> 2))) & 3) << 4);
        const int l2 = (i2 < 8 ? s[i2] & 0xF : s[i2 - 8] >> 4) | (((s[8 + (i2 & 3)] >> (2 * (i2 >> 2))) & 3) << 4);
        const float d = rdh(b + 108);
        const int hs = 4 * n + j;
        for (int w = 0; w < 8; ++w) {
            bytes2(((qq[w] >> (2 * j)) & 0x0303) | (((hm[w] >> hs) & 0x0101) << 2), q + 2 * w);
            bytes2(((qq[w + 8] >> (2 * j)) & 0x0303) | (((hm[w + 8] >> hs) & 0x0101) << 2), q + 16 + 2 * w);
        }
        const float d1 = d * float(l1 - 32), d2 = d * float(l2 - 32);
        sc = float4(d1, d2, 4.0f * d1, 4.0f * d2);
    }
};

template <> struct Dec<T_F32> {   // 16 floats a unit (the Unsloth files' router / alpha / beta / inject rows)
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const float4* p = (device const float4*)(row + 64 * u);
        for (int k = 0; k < 4; ++k) {
            const float4 v = p[k];
            q[4 * k] = v.x; q[4 * k + 1] = v.y; q[4 * k + 2] = v.z; q[4 * k + 3] = v.w;
        }
        sc = float4(1.0f, 1.0f, 0.0f, 0.0f);
    }
};

template <> struct Dec<T_BF16> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uint4* p = (device const uint4*)(row + 32 * u);
        const uint4 a = p[0], c = p[1];
        const uint v[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
        for (int k = 0; k < 8; ++k) {
            q[2 * k] = as_type<float>(v[k] << 16);
            q[2 * k + 1] = as_type<float>(v[k] & 0xFFFF0000u);
        }
        sc = float4(1.0f, 1.0f, 0.0f, 0.0f);
    }
};

// acc += unit dot of decoded q / sc with one token's prepared slice
template <int UNIT>
static inline float unit_combine(thread const float* q, float4 sc, thread const float* xs, thread const float* xsum) {
    constexpr int H = UNIT / 2;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int j = 0; j < H; j += 2) {
        s0 = fma(q[j], xs[j], s0);
        s1 = fma(q[j + 1], xs[j + 1], s1);
        s2 = fma(q[H + j], xs[H + j], s2);
        s3 = fma(q[H + j + 1], xs[H + j + 1], s3);
    }
    return sc.x * (s0 + s1) + sc.y * (s2 + s3) - (sc.z * xsum[0] + sc.w * xsum[1]);
}
