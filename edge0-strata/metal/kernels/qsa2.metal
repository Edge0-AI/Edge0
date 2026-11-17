// metal/kernels/qsa2.metal - QSA decode v2 (after qsa.metal): the same selection and attention math, more parallel.
//
//   qsa_attn2   chunks of 32 cells, the 12 query heads of a kv head in 2 groups of 6: 4x the threadgroups of
//               qsa_attn (at a 1K context 128 instead of 32); V loads 8 cells at a time.
//   qsa_merge2  qsa_merge for that chunking.
//   qsa_topk2   the digit of each radix pass found by a parallel suffix scan of the 256 bins (simdgroup prefix
//               sums) instead of one thread; cells emitted from contiguous block ranges per thread after one scan.

#define QCH2 32
#define QHG 2

// partials: [n_tok][chunks2][QH][QD + 2].  grid (chunks2, QKV * QHG, n_tok), 256 threads.
kernel void qsa_attn2(device const float* qcur [[buffer(0)]], device const char* kq [[buffer(1)]],
                      device const char* vq [[buffer(2)]], device const half* ksc [[buffer(3)]],
                      device const half* vsc [[buffer(4)]], device const int* ids [[buffer(5)]],
                      device float* part [[buffer(6)]], constant QsaArgs& a [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HPG = QH / QKV / QHG;   // 6
    threadgroup float qs[HPG][QD];
    threadgroup float s[HPG][QCH2];
    threadgroup float mh[HPG], lh[HPG];
    threadgroup int cid[QCH2];
    const int chunk = int(tg.x), g = int(tg.y) / QHG, hg = int(tg.y) % QHG, t = int(tg.z);
    const int h0 = g * (QH / QKV) + hg * HPG;
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH2;
    const int nchunks = (QBUDGET + QCH2 - 1) / QCH2;
    if (uint(c0) >= width) return;
    device float* pp = part + ((uint(t) * nchunks + chunk) * QH) * (QD + 2);
    const int nc = min(QCH2, int(width) - c0);
    for (int h = 0; h < HPG; ++h) qs[h][tid] = qcur[(t * QH + h0 + h) * QD + tid];
    if (tid < QCH2) cid[tid] = tid < nc ? ids[t * QBUDGET + c0 + tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = sg; c < nc; c += 8) {
        const uint rowi = uint(cid[c]) * QKV + uint(g);
        device const char4* kr = (device const char4*)(kq + rowi * QD + lane * 8);
        const char4 k0 = kr[0], k1 = kr[1];
        const float scl = float(ksc[rowi * 4 + (lane >> 3)]);
        const float kv[8] = {float(k0.x) * scl, float(k0.y) * scl, float(k0.z) * scl, float(k0.w) * scl,
                             float(k1.x) * scl, float(k1.y) * scl, float(k1.z) * scl, float(k1.w) * scl};
        for (int h = 0; h < HPG; ++h) {
            float d = 0.0f;
            for (int i = 0; i < 8; ++i) d = fma(qs[h][lane * 8 + i], kv[i], d);
            d = simd_sum(d);
            if (lane == 0) s[h][c] = d * 0.0625f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < HPG) {
        const int h = sg;
        const float sv = lane < nc ? s[h][lane] : -INFINITY;
        const float m = simd_max(sv);
        const float p = lane < nc ? exp(sv - m) : 0.0f;
        const float l = simd_sum(p);
        if (lane < nc) s[h][lane] = p;
        if (lane == 0) { mh[h] = m; lh[h] = l; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[HPG];
    for (int h = 0; h < HPG; ++h) acc[h] = 0.0f;
    for (int cb = 0; cb < nc; cb += 8) {
        float v[8];
        const int m = min(8, nc - cb);
        for (int c = 0; c < 8; ++c) {
            v[c] = 0.0f;
            if (c < m) {
                const uint rowi = uint(cid[cb + c]) * QKV + uint(g);
                v[c] = float(vq[rowi * QD + tid]) * float(vsc[rowi * 4 + (tid >> 6)]);
            }
        }
        for (int c = 0; c < m; ++c)
            for (int h = 0; h < HPG; ++h) acc[h] = fma(s[h][cb + c], v[c], acc[h]);
    }
    for (int h = 0; h < HPG; ++h) {
        device float* o = pp + (h0 + h) * (QD + 2);
        o[tid] = acc[h];
        if (tid == 0) { o[QD] = mh[h]; o[QD + 1] = lh[h]; }
    }
}

kernel void qsa_merge2(device const float* part [[buffer(0)]], device const float* qfull [[buffer(1)]],
                       device float* xp [[buffer(2)]], constant QsaArgs& a [[buffer(3)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float msum[256];
    const int h = int(tg.x), t = int(tg.y);
    const int nchunks = (QBUDGET + QCH2 - 1) / QCH2;
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int nc = int((width + QCH2 - 1) / QCH2);
    device const float* pp = part + (uint(t) * nchunks * QH + uint(h)) * (QD + 2);
    float M = -INFINITY;
    for (int k = 0; k < nc; ++k) M = max(M, pp[k * QH * (QD + 2) + QD]);
    float L = 0.0f, o = 0.0f;
    for (int k = 0; k < nc; ++k) {
        device const float* q = pp + k * QH * (QD + 2);
        const float e = exp(q[QD] - M);
        L = fma(q[QD + 1], e, L);
        o = fma(q[tid], e, o);
    }
    o = o / L;
    const float gate = qfull[(t * QH + h) * 2 * QD + QD + tid];
    const float y = o * (1.0f / (1.0f + exp(-gate)));
    const uint n = QH * QD, plane = a.n_tok * n, d = uint(h) * QD + tid;
    const int j = tid & 15;
    const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
    const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
    device float* raw = xp + uint(t) * n + d;
    raw[0] = y;
    raw[plane] = y * q2s[j & 7];
    raw[2 * plane] = y * p4s[j & 3];
    raw[3 * plane] = (j & 1) ? y * 0.00390625f : y;
    msum[tid] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0) {
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += msum[tid + k]; s1 += msum[tid + k + 1]; }
        xp[4 * plane + uint(t) * (n / 16) + d / 16] = s0 + s1;
    }
}

kernel void qsa_topk2(device const float* scores [[buffer(0)]], device int* ids [[buffer(1)]],
                      constant QsaArgs& a [[buffer(2)]],
                      device const uint* rinfo [[buffer(3), function_constant(qsa_rows)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint sgs[32], sgs2[32];
    threadgroup uint sel[2];
    const int t = int(tg.x);
    const uint n_kv = (qsa_rows ? rinfo[8 + 4 * t + 1] : a.pos0 + uint(t)) + 1, n_bid = n_kv >> 2, nb = n_bid + 1;
    device int* out = ids + t * QBUDGET;
    device const float* sc = scores + uint(t) * (a.max_cells / 4 + 2);
    if (n_kv <= QBUDGET) {
        for (uint c = tid; c < n_kv; c += 1024) out[c] = int(c);
        return;
    }
    uint prefix = 0, pmask = 0, need = QBUDGET;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = tid; b < nb; b += 1024) {
            const uint k = okey(sc[b]);
            if ((k & pmask) == prefix) {
                const uint w = b < n_bid ? 4u : (n_kv & 3u);
                if (w) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], w, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // thread i < 256 holds bin 255 - i: an exclusive prefix over i = the cells of the higher digits
        uint cnt = 0, ex = 0;
        if (tid < 256) {
            cnt = atomic_load_explicit(&hist[255 - tid], memory_order_relaxed);
            ex = simd_prefix_exclusive_sum(cnt);
            if (lane == 31) sgs[sg] = ex + cnt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256) {
            uint above = ex;
            for (int i = 0; i < int(sg); ++i) above += sgs[i];
            if (above < need && above + cnt >= need) { sel[0] = 255u - tid; sel[1] = need - above; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= sel[0] << shift;
        pmask |= 255u << shift;
        need = sel[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint T = prefix;
    const uint per = (nb + 1023) / 1024, b0 = min(nb, uint(tid) * per), b1 = min(nb, b0 + per);
    uint gt = 0, eq = 0;
    for (uint b = b0; b < b1; ++b) {
        const uint k = okey(sc[b]);
        const uint cells = b < n_bid ? 4u : (n_kv & 3u);
        if (k > T) gt += cells;
        else if (k == T) eq += cells;
    }
    const uint eqb = simd_prefix_exclusive_sum(eq);
    if (lane == 31) sgs2[sg] = eqb + eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint eq_before = eqb;
    for (int i = 0; i < int(sg); ++i) eq_before += sgs2[i];
    const uint eq_take = eq_before >= need ? 0u : min(eq, need - eq_before);
    const uint mine = gt + eq_take;
    const uint mb = simd_prefix_exclusive_sum(mine);
    if (lane == 31) sgs[sg] = mb + mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint off = mb;
    for (int i = 0; i < int(sg); ++i) off += sgs[i];
    uint left = eq_take;
    for (uint b = b0; b < b1; ++b) {
        const uint k = okey(sc[b]);
        const uint cells = b < n_bid ? 4u : (n_kv & 3u);
        uint take = 0;
        if (k > T) take = cells;
        else if (k == T) { take = min(cells, left); left -= take; }
        for (uint c = 0; c < take; ++c) out[off + c] = int(4 * b + c);
        off += take;
    }
}

// qsa_attn3: qsa_attn's chunking and partial layout (qsa_merge reads it), without a reduction per (cell, head):
// for q.k a lane owns a cell (8 simdgroups = 2 halves of the 64 cells x 4 groups of 3 heads; q from threadgroup
// memory, read by every lane alike), for p.v a thread owns a dimension.  grid (33, QKV, n_tok), 256 threads.
kernel void qsa_attn3(device const float* qcur [[buffer(0)]], device const char* kq [[buffer(1)]],
                      device const char* vq [[buffer(2)]], device const half* ksc [[buffer(3)]],
                      device const half* vsc [[buffer(4)]], device const int* ids [[buffer(5)]],
                      device float* part [[buffer(6)]], constant QsaArgs& a [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HPG = QH / QKV;   // 12
    threadgroup float qs[HPG][QD];
    threadgroup float s[HPG][QCH];
    threadgroup float mh[HPG], lh[HPG];
    threadgroup int cid[QCH];
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH;
    const int nchunks = (QBUDGET + QCH - 1) / QCH;
    device float* pp = part + ((uint(t) * nchunks + chunk) * QH) * (QD + 2);
    if (uint(c0) >= width) return;
    const int nc = min(QCH, int(width) - c0);
    for (int h = 0; h < HPG; ++h) qs[h][tid] = qcur[(t * QH + g * HPG + h) * QD + tid];
    if (tid < QCH) cid[tid] = tid < nc ? ids[t * QBUDGET + c0 + tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int half_ = sg & 1, hgrp = sg >> 1;     // cells 32 * half_ + lane, heads 3 * hgrp .. + 3
        const int c = 32 * half_ + lane;
        if (c < nc) {
            const uint rowi = uint(cid[c]) * QKV + uint(g);
            device const char4* kr = (device const char4*)(kq + rowi * QD);
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            for (int gq = 0; gq < 4; ++gq) {
                const float scl = float(ksc[rowi * 4 + gq]);
                float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    const char4 k4 = kr[gq * 16 + i];
                    const int base = gq * 64 + 4 * i;
                    const float kv[4] = {float(k4.x), float(k4.y), float(k4.z), float(k4.w)};
                    for (int e = 0; e < 4; ++e) {
                        e0 = fma(qs[3 * hgrp][base + e], kv[e], e0);
                        e1 = fma(qs[3 * hgrp + 1][base + e], kv[e], e1);
                        e2 = fma(qs[3 * hgrp + 2][base + e], kv[e], e2);
                    }
                }
                d0 = fma(e0, scl, d0);
                d1 = fma(e1, scl, d1);
                d2 = fma(e2, scl, d2);
            }
            s[3 * hgrp][c] = d0 * 0.0625f;
            s[3 * hgrp + 1][c] = d1 * 0.0625f;
            s[3 * hgrp + 2][c] = d2 * 0.0625f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int h = sg; h < HPG; h += 8) {
        float m = -INFINITY;
        for (int c = lane; c < nc; c += 32) m = max(m, s[h][c]);
        m = simd_max(m);
        float l = 0.0f;
        for (int c = lane; c < nc; c += 32) {
            const float p = exp(s[h][c] - m);
            s[h][c] = p;
            l += p;
        }
        l = simd_sum(l);
        if (lane == 0) { mh[h] = m; lh[h] = l; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[HPG];
    for (int h = 0; h < HPG; ++h) acc[h] = 0.0f;
    for (int cb = 0; cb < nc; cb += 8) {
        float v[8];
        const int m = min(8, nc - cb);
        for (int c = 0; c < 8; ++c) {
            v[c] = 0.0f;
            if (c < m) {
                const uint rowi = uint(cid[cb + c]) * QKV + uint(g);
                v[c] = float(vq[rowi * QD + tid]) * float(vsc[rowi * 4 + (tid >> 6)]);
            }
        }
        for (int c = 0; c < m; ++c)
            for (int h = 0; h < HPG; ++h) acc[h] = fma(s[h][cb + c], v[c], acc[h]);
    }
    for (int h = 0; h < HPG; ++h) {
        device float* o = pp + (g * HPG + h) * (QD + 2);
        o[tid] = acc[h];
        if (tid == 0) { o[QD] = mh[h]; o[QD + 1] = lh[h]; }
    }
}
