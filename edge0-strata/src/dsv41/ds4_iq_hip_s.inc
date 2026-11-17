// Strata. Part of ds4_iq_hip_r.inc: the weights of a whole 64-value step (two 32-value sub-blocks) of one weight row in a few wide
// loads, and the decode of its four 8-value groups (k-steps) from registers.  The loads are unaligned (the blocks of the i-quants are
// 2-byte aligned; the AMD hardware takes unaligned vector loads).  A lane holds the group p = 2 ks + hi of every sub-block, so
// the two lanes of a column (hi 0 / 1) share most of their bytes.  s: the step (64 values), b: the block (256 values: 4 steps),
// jb = 2 (s & 3): the first sub-block of the step in its block.
struct SRaw {
    uint32_t w[10];
};
__device__ __forceinline__ uint4 ldu128(const void* p) {
    uint4 v;
    __builtin_memcpy(&v, p, 16);
    return v;
}
__device__ __forceinline__ uint2 ldu64(const void* p) {
    uint2 v;
    __builtin_memcpy(&v, p, 8);
    return v;
}
__device__ __forceinline__ uint32_t ldu32(const void* p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}
__device__ __forceinline__ uint32_t ldu16(const void* p) {
    uint16_t v;
    __builtin_memcpy(&v, p, 2);
    return v;
}
template <int T>
__device__ __forceinline__ void stage_load(const uint8_t* __restrict__ row, int s, int hi, SRaw& r) {
    if constexpr (T == kIQ4_NL) {   // 32-value blocks of 18 bytes: the step is blocks 2 s and 2 s + 1
        const uint8_t* b0 = row + (size_t) (2 * s) * 18;
        const uint2 qa = ldu64(b0 + 2 + 8 * hi), qb = ldu64(b0 + 18 + 2 + 8 * hi);
        r.w[0] = qa.x, r.w[1] = qa.y, r.w[2] = qb.x, r.w[3] = qb.y;
        r.w[4] = ldu16(b0);
        r.w[5] = ldu16(b0 + 18);
    } else {
        const uint8_t* b = row + (size_t) (s >> 2) * Ty<T>::B;
        const int jb = 2 * (s & 3);
        if constexpr (T == kIQ2_XXS) {
            const uint4 q = ldu128(b + 2 + 8 * jb);
            r.w[0] = q.x, r.w[1] = q.y, r.w[2] = q.z, r.w[3] = q.w;
            r.w[4] = ldu16(b);
        } else if constexpr (T == kIQ2_XS) {
            const uint4 q = ldu128(b + 2 + 8 * jb);
            r.w[0] = q.x, r.w[1] = q.y, r.w[2] = q.z, r.w[3] = q.w;
            r.w[4] = ldu16(b + 66 + jb);
            r.w[5] = ldu16(b);
        } else if constexpr (T == kIQ2_S) {
            const uint2 q = ldu64(b + 2 + 4 * jb), sg = ldu64(b + 34 + 4 * jb);
            r.w[0] = q.x, r.w[1] = q.y, r.w[2] = sg.x, r.w[3] = sg.y;
            r.w[4] = ldu16(b + 66 + jb);
            r.w[5] = ldu16(b + 74 + jb);
            r.w[6] = ldu16(b);
        } else if constexpr (T == kIQ3_XXS) {
            const uint4 q = ldu128(b + 2 + 8 * jb);
            const uint2 ax = ldu64(b + 66 + 4 * jb);
            r.w[0] = q.x, r.w[1] = q.y, r.w[2] = q.z, r.w[3] = q.w;
            r.w[4] = ax.x, r.w[5] = ax.y;
            r.w[6] = ldu16(b);
        } else if constexpr (T == kIQ3_S) {
            const uint4 q = ldu128(b + 2 + 8 * jb);
            const uint2 sg = ldu64(b + 74 + 4 * jb);
            r.w[0] = q.x, r.w[1] = q.y, r.w[2] = q.z, r.w[3] = q.w;
            r.w[4] = ldu16(b + 66 + jb);
            r.w[5] = sg.x, r.w[6] = sg.y;
            r.w[7] = b[106 + (jb >> 1)];
            r.w[8] = ldu16(b);
        } else if constexpr (T == kIQ4_XS) {
            const uint2 qa = ldu64(b + 8 + 16 * jb + 8 * hi), qb = ldu64(b + 8 + 16 * (jb + 1) + 8 * hi);
            r.w[0] = qa.x, r.w[1] = qa.y, r.w[2] = qb.x, r.w[3] = qb.y;
            r.w[4] = ldu16(b + 2);
            r.w[5] = b[4 + (jb >> 1)];
            r.w[6] = ldu16(b);
        } else if constexpr (T == kIQ1_S) {
            const uint2 q = ldu64(b + 2 + 4 * jb);
            r.w[0] = q.x, r.w[1] = q.y;
            r.w[2] = ldu32(b + 34 + 2 * jb);
            r.w[3] = ldu16(b);
        } else {   // IQ1_M
            const uint2 q = ldu64(b + 4 * jb);
            const uint2 d4 = ldu64(b + 48);
            r.w[0] = q.x, r.w[1] = q.y;
            r.w[2] = ldu32(b + 32 + 2 * jb);
            r.w[3] = ldu16(b + 48 + (jb >> 1) * 2);
            r.w[4] = d4.x, r.w[5] = d4.y;
        }
    }
}
// the k-step t (0..3, compile time) of a loaded step: sub-block sub = t >> 1, group p = 2 (t & 1) + hi
template <int T, int TT>
__device__ __forceinline__ void step_dec(const SRaw& r, int s, int hi, const uint32_t* tab, int& c0, int& c1, float& sc) {
    constexpr int sub = TT >> 1, ks = TT & 1;
    const int p = 2 * ks + hi;
    if constexpr (T == kIQ2_XXS) {
        const uint2 g = ((const uint2*) tab)[(r.w[2 * sub] >> (8 * p)) & 255u];
        const uint32_t a1 = r.w[2 * sub + 1];
        const uint32_t sg = ksigns8(a1 >> (7 * p));
        c0 = sgn4(g.x, sg);
        c1 = sgn4(g.y, sg >> 4);
        sc = hf(r.w[4]) * (float) (2 * (int) (a1 >> 28) + 1) * 0.125f;
    } else if constexpr (T == kIQ2_XS) {
        const uint32_t v = (r.w[2 * sub + (ks)] >> (16 * hi)) & 0xffffu;
        const uint2 g = ((const uint2*) tab)[v & 511u];
        const uint32_t sg = ksigns8(v >> 9);
        const uint32_t scb = (r.w[4] >> (8 * sub)) & 255u;
        c0 = sgn4(g.x, sg);
        c1 = sgn4(g.y, sg >> 4);
        sc = hf(r.w[5]) * (float) (2 * (int) (ks == 0 ? scb & 15 : scb >> 4) + 1) * 0.125f;
    } else if constexpr (T == kIQ2_S) {
        const uint32_t qh = (r.w[4] >> (8 * sub)) & 255u;
        const uint32_t idx = ((r.w[sub] >> (8 * p)) & 255u) | ((qh << (8 - 2 * p)) & 0x300u);
        const uint32_t sg = (r.w[2 + sub] >> (8 * p)) & 255u;
        const uint32_t scb = (r.w[5] >> (8 * sub)) & 255u;
        const uint2 g = ((const uint2*) tab)[idx];
        c0 = sgn4(g.x, sg);
        c1 = sgn4(g.y, sg >> 4);
        sc = hf(r.w[6]) * (float) (2 * (int) (ks == 0 ? scb & 15 : scb >> 4) + 1) * 0.125f;
    } else if constexpr (T == kIQ3_XXS) {
        const uint32_t word = r.w[2 * sub + ks];
        const uint32_t aux = r.w[4 + sub];
        const uint32_t sg = ksigns8(aux >> (7 * p));
        c0 = sgn4(tab[(word >> (16 * hi)) & 255u], sg);
        c1 = sgn4(tab[(word >> (16 * hi + 8)) & 255u], sg >> 4);
        sc = hf(r.w[6]) * (float) (2 * (int) (aux >> 28) + 1) * 0.25f;
    } else if constexpr (T == kIQ3_S) {
        const uint32_t word = r.w[2 * sub + ks];
        const uint32_t h = (r.w[4] >> (8 * sub)) & 255u;
        const uint32_t sg = (r.w[5 + sub] >> (8 * p)) & 255u;
        c0 = sgn4(tab[((word >> (16 * hi)) & 255u) | ((h << (8 - 2 * p)) & 256u)], sg);
        c1 = sgn4(tab[((word >> (16 * hi + 8)) & 255u) | ((h << (7 - 2 * p)) & 256u)], sg >> 4);
        sc = hf(r.w[8]) * (float) (1 + 2 * (int) ((r.w[7] >> (4 * sub)) & 15));
    } else if constexpr (T == kIQ4_XS) {
        const int jb = 2 * (s & 3);
        const int ls = ((r.w[5] >> (4 * sub)) & 15) | (((r.w[4] >> (2 * (jb + sub))) & 3) << 4);
        const int2 v0 = tab16(r.w[2 * sub], tab), v1 = tab16(r.w[2 * sub + 1], tab);
        c0 = ks == 0 ? v0.x : v0.y;
        c1 = ks == 0 ? v1.x : v1.y;
        sc = hf(r.w[6]) * (float) (ls - 32);
    } else if constexpr (T == kIQ1_S) {
        const uint32_t qh = (r.w[2] >> (16 * sub)) & 0xffffu;
        const uint32_t g = tab[((r.w[sub] >> (8 * p)) & 255u) | (((qh >> (3 * p)) & 7u) << 8)];
        const bool neg = (qh & 0x8000u) != 0;
        c0 = (int) iq1_codes(g & 0x0F0F0F0Fu, neg);
        c1 = (int) iq1_codes((g >> 4) & 0x0F0F0F0Fu, neg);
        sc = hf(r.w[3]) * (float) (2 * (int) ((qh >> 12) & 7u) + 1) * 0.125f;
    } else if constexpr (T == kIQ1_M) {
        const uint32_t qh = (r.w[2] >> (8 * (2 * sub + ks))) & 255u;   // (p < 2: the first byte of the sub-block, else the second)
        const uint32_t g = tab[((r.w[sub] >> (8 * p)) & 255u) | ((qh << (hi ? 4 : 8)) & 0x700u)];
        const bool neg = (hi ? qh & 0x80u : qh & 0x08u) != 0;
        c0 = (int) iq1_codes(g & 0x0F0F0F0Fu, neg);
        c1 = (int) iq1_codes((g >> 4) & 0x0F0F0F0Fu, neg);
        const int sh = (int) ((r.w[3] >> (6 * sub + (ks == 0 ? 0 : 3))) & 7u);
        const uint32_t d16 = (r.w[4] >> 12 & 0xfu) | ((r.w[4] >> 24) & 0xf0u) | ((r.w[5] >> 4) & 0xf00u) | ((r.w[5] >> 16) & 0xf000u);
        sc = hf(d16) * (float) (2 * sh + 1) * 0.125f;
    } else {   // IQ4_NL
        const int2 v0 = tab16(r.w[2 * sub], tab), v1 = tab16(r.w[2 * sub + 1], tab);
        c0 = ks == 0 ? v0.x : v0.y;
        c1 = ks == 0 ? v1.x : v1.y;
        sc = hf(r.w[4 + sub]);
    }
}
