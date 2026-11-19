// metal/kernels/mma.metal - a GEMV for windows of tokens on the simdgroup matrix units (after quant/units.metal).
//
// y[t][r] = W[r] . x[t] for NTOK (1..8) tokens: each simdgroup owns 8 rows; the K axis runs in chunks of 128.  Per
// chunk the simdgroup decodes its 8 x 128 weights - with ref_quant's arithmetic, so a tile element is exactly the
// dequantized weight - into threadgroup memory, then 16 simdgroup_multiply_accumulate steps take A = the 8 x 8 weight
// tile and B = the 8 (k) x 8 (token) slice of x, loaded transposed straight from device memory (x has 8 rows: the
// tokens past the window are padding, and an output column depends only on its own token).
//
// Activations are the raw plane (no pre-scaling); n_in must be a multiple of 128.  Every format's units sit inside
// one 128-wide chunk: Q4_K/Q5_K units span 64 positions, Q6_K units 128, the others are contiguous.

// ---- decoders to actual weights: w[UNIT] and the chunk offsets of its two halves (p0: w[0..H), p1: w[H..UNIT)) ----
template <int F> struct DW {
    // generic: w = s * q (quant.metal's deq is ref_quant's order: the scale times the code), contiguous
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint* tab) {
        constexpr int UNIT = fmt<F>::UNIT, UPB = fmt<F>::QK / UNIT;
        float q[UNIT];
        const float s = deq<F>(row + (u / UPB) * fmt<F>::BB, u % UPB, q, tab);
        for (int j = 0; j < UNIT; ++j) w[j] = q[j] * s;
        p0 = u * UNIT - k0;
        p1 = p0 + UNIT / 2;
    }
};
// Q2_0: (code - 1) * d
template <> struct DW<T_Q2_0> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint*) {
        device const uchar* b = row + (u >> 2) * 18;
        device const ushort* p = (device const ushort*)(b + 2 + 4 * (u & 3));
        const ushort a = p[0], c = p[1];
        const float d = rdh(b);
        for (int j = 0; j < 8; ++j) {
            w[j] = float(int((a >> (2 * j)) & 3) - 1) * d;
            w[8 + j] = float(int((c >> (2 * j)) & 3) - 1) * d;
        }
        p0 = 16 * u - k0;
        p1 = p0 + 8;
    }
};
template <> struct DW<T_Q5_1> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint* tab) {
        device const uchar* b = row + u * 24;
        float q[32];
        const float d = deq<T_Q5_1>(b, 0, q, tab), m = rdh(b + 2);
        for (int j = 0; j < 32; ++j) w[j] = q[j] * d + m;
        p0 = 32 * u - k0;
        p1 = p0 + 16;
    }
};
template <> struct DW<T_F32> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint*) {
        device const float* p = (device const float*)(row + 64 * u);
        for (int j = 0; j < 16; ++j) w[j] = p[j];
        p0 = 16 * u - k0;
        p1 = p0 + 8;
    }
};
template <> struct DW<T_BF16> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint*) {
        device const uint4* p = (device const uint4*)(row + 32 * u);
        const uint4 a = p[0], c = p[1];
        const uint v[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
        for (int k = 0; k < 8; ++k) {
            w[2 * k] = as_type<float>(v[k] << 16);
            w[2 * k + 1] = as_type<float>(v[k] & 0xFFFF0000u);
        }
        p0 = 16 * u - k0;
        p1 = p0 + 8;
    }
};
// Q4_K / Q5_K: d1 * code - m1 (ref: d * sc, dmin * m), runs at 256b + 64c + 16m and + 32
template <> struct DW<T_Q4_K> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 144;
        const int c = (u >> 1) & 3, m = u & 1;
        const float d = rdh(b), dm = rdh(b + 2);
        float s1, m1, s2, m2;
        scale_min_k4(2 * c, b + 4, s1, m1);
        scale_min_k4(2 * c + 1, b + 4, s2, m2);
        const float d1 = d * s1, mm1 = dm * m1, d2 = d * s2, mm2 = dm * m2;
        device const uchar* q = b + 16 + 32 * c + 16 * m;
        for (int j = 0; j < 16; ++j) {
            w[j] = d1 * float(q[j] & 15) - mm1;
            w[16 + j] = d2 * float(q[j] >> 4) - mm2;
        }
        p0 = 256 * (u >> 3) + 64 * c + 16 * m - k0;
        p1 = p0 + 32;
    }
};
template <> struct DW<T_Q5_K> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 176;
        const int c = (u >> 1) & 3, m = u & 1;
        const float d = rdh(b), dm = rdh(b + 2);
        float s1, m1, s2, m2;
        scale_min_k4(2 * c, b + 4, s1, m1);
        scale_min_k4(2 * c + 1, b + 4, s2, m2);
        const float d1 = d * s1, mm1 = dm * m1, d2 = d * s2, mm2 = dm * m2;
        device const uchar* ql = b + 48 + 32 * c + 16 * m;
        device const uchar* qh = b + 16 + 16 * m;
        const uchar u1 = uchar(1 << (2 * c)), u2 = uchar(2 << (2 * c));
        for (int j = 0; j < 16; ++j) {
            w[j] = d1 * float((ql[j] & 15) + ((qh[j] & u1) ? 16 : 0)) - mm1;
            w[16 + j] = d2 * float((ql[j] >> 4) + ((qh[j] & u2) ? 16 : 0)) - mm2;
        }
        p0 = 256 * (u >> 3) + 64 * c + 16 * m - k0;
        p1 = p0 + 32;
    }
};
// Q6_K: d * sc * (code - 32), runs at 256b + 128n + 32p + 16m and + 64
template <> struct DW<T_Q6_K> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 210;
        const int n = (u >> 2) & 1, m = (u >> 1) & 1, p = u & 1;
        device const uchar* ql = b + 64 * n + 32 * p + 16 * m;
        device const uchar* qh = b + 128 + 32 * n + 16 * m;
        device const char* sc = (device const char*)(b + 192);
        const float d = rdh(b + 208);
        const float ds1 = d * float(sc[8 * n + m + 2 * p]), ds2 = d * float(sc[8 * n + m + 2 * p + 4]);
        for (int j = 0; j < 16; ++j) {
            const int c1 = int((ql[j] & 15) | (((qh[j] >> (2 * p)) & 3) << 4)) - 32;
            const int c2 = int((ql[j] >> 4) | (((qh[j] >> (2 * p + 4)) & 3) << 4)) - 32;
            w[j] = ds1 * float(c1);
            w[16 + j] = ds2 * float(c2);
        }
        p0 = 256 * (u >> 3) + 128 * n + 32 * p + 16 * m - k0;
        p1 = p0 + 64;
    }
};
// Q3_K: d_all * (sc - 32) * (code - (hbit ? 0 : 4)), contiguous 32
template <> struct DW<T_Q3_K> {
    static inline void run(device const uchar* row, int u, int k0, thread float* w, thread int& p0, thread int& p1,
                           threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 110;
        const int n = (u >> 2) & 1, j = u & 3;
        device const uchar* q = b + 32 + 32 * n;
        device const uchar* hm = b;
        device const uchar* s = b + 96;
        const int i1 = 8 * n + 2 * j, i2 = i1 + 1;
        const int l1 = (i1 < 8 ? s[i1] & 0xF : s[i1 - 8] >> 4) | (((s[8 + (i1 & 3)] >> (2 * (i1 >> 2))) & 3) << 4);
        const int l2 = (i2 < 8 ? s[i2] & 0xF : s[i2 - 8] >> 4) | (((s[8 + (i2 & 3)] >> (2 * (i2 >> 2))) & 3) << 4);
        const float d = rdh(b + 108);
        const float dl1 = d * float(l1 - 32), dl2 = d * float(l2 - 32);
        const uchar hb = uchar(1 << (4 * n + j));
        for (int l = 0; l < 16; ++l) {
            w[l] = dl1 * float(int((q[l] >> (2 * j)) & 3) - ((hm[l] & hb) ? 0 : 4));
            w[16 + l] = dl2 * float(int((q[l + 16] >> (2 * j)) & 3) - ((hm[l + 16] & hb) ? 0 : 4));
        }
        p0 = 32 * u - k0;
        p1 = p0 + 16;
    }
};

struct MmaArgs {
    uint n_in;
    uint n_out;
    uint row_bytes;
    uint x_stride;   // floats between the 8 token rows of x
    uint y_stride;
};
constant int MMA_NTOK [[function_constant(4)]];

template <int F, int SG>
kernel void gemv_mma(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* y [[buffer(2)]], constant MmaArgs& a [[buffer(3)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, UPC = 128 / UNIT, H = UNIT / 2;
    threadgroup float wt[SG][8 * 128];
    threadgroup float ct[SG][64];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int row0 = (int(tg.x) * SG + sg) * 8;
    simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    threadgroup float* w8 = wt[sg];
    for (int k0 = 0; k0 < int(a.n_in); k0 += 128) {
        for (int it = lane; it < 8 * UPC; it += 32) {
            const int r = it / UPC, uu = it - r * UPC;
            const int rr = min(row0 + r, int(a.n_out) - 1);
            float w[UNIT];
            int p0, p1;
            DW<F>::run(W + ulong(rr) * a.row_bytes, k0 / UNIT + uu, k0, w, p0, p1, tab);
            for (int j = 0; j < H; ++j) {
                w8[r * 128 + p0 + j] = w[j];
                w8[r * 128 + p1 + j] = w[H + j];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int k8 = 0; k8 < 128; k8 += 8) {
            simdgroup_float8x8 A, B;
            simdgroup_load(A, w8 + k8, 128);
            simdgroup_load(B, x + k0 + k8, a.x_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(C, ct[sg], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 64; e += 32) {
        const int r = e >> 3, t = e & 7;
        if (t < MMA_NTOK && row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = ct[sg][e];
    }
}

#define MM(F, S) template [[host_name("gemv_mma_" #F "_" #S)]] kernel decltype(gemv_mma<F, S>) gemv_mma<F, S>;
#define MM_SET(F) MM(F, 1) MM(F, 2) MM(F, 4)
MM_SET(T_BF16) MM_SET(T_Q8_0) MM_SET(T_Q4_0) MM_SET(T_Q5_0) MM_SET(T_IQ4_NL) MM_SET(T_IQ4_XS) MM_SET(T_Q2_0)
MM_SET(T_IQ3_S) MM_SET(T_Q3_K) MM_SET(T_Q4_K) MM_SET(T_Q5_K) MM_SET(T_Q6_K)
