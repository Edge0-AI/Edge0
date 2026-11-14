// metal/kernels/qsa.metal - QSA (sparse full attention) decode for a window of tokens, between the projections.
// Contract of include/strata/kernels/qsa.hpp / qsa_select.hpp / qsa_decode_attn.hpp / kv_q8.hpp (the engine's
// native path), in a Metal layout (no paging: unified memory):
//
//   qsa_prep    per token t (cell pos0 + t): q heads rms_norm * q_norm, NEOX RoPE on 64 dims -> qcur;
//               k rms_norm * k_norm, RoPE -> int8 KV append (fp16 scale per 64 = fp16(amax/127), rint, clamp);
//               v -> int8 append; indexer q rms_norm * iq_norm, RoPE -> qidx.  The indexer KEYS (last
//               threadgroup, all window tokens in order): raw tail [cell % 4]; on a block completion
//               pooled[b] = rope(rms_norm(mean of the block's 4 raw keys) * ik_norm, pos of the block's first cell);
//               cell 0 also sets dead = rms_norm(raw) * ik_norm (the tail block's key).
//   qsa_score   per token, block b <= n_bid: sum over the 4 indexer heads of relu(qidx . key_b); key_b = pooled[b],
//               the tail block n_bid scores `dead` and gets +1e9 when it has cells.
//   qsa_topk    per token: width = min(n_kv, 2051) cells; n_kv <= 2051: all; else the width cells of largest
//               score (a cell inherits its block's), ties to the lowest cell, emitted ascending (radix select).
//   qsa_attn    split-K: (chunk of CH cells, kv head, token): 12 query heads; s = q.k / 16, chunk max m, sum l,
//               acc = p . v  -> partials
//   qsa_merge   per (head, token): log-sum-exp merge, * sigmoid(gate), written as prepared activations (xprep
//               planes) for attn_output.
//
// RoPE: cos/sin from a host table [pos][32] (double precision, as build_rope_table).

#define QH 24
#define QKV 2
#define QD 256
#define QROT 64
#define IH 4
#define ID 128
#define QBUDGET 2051
#define QCH 64

constant int QSA_NTOK [[function_constant(3)]];
// QSA_MROPE (function constant 24): interleaved M-RoPE (the vision path; strata/kernels/mrope.hpp, ggml IMROPE,
// sections 11/11/10): buffer 20 of qsa_prep is a table int32 [cells][3] (t, h, w), and rotary pair i (0..31) of the row
// at (pos_base + cell) rotates by the cos/sin table's row tab[(pos_base + cell) * 3 + i % 3] instead of its own -
// the pooled indexer keys too (position pos_base + 4 b of a block).  Off: exactly the plain RoPE.
constant bool QSA_MROPE [[function_constant(24)]];
constant bool qsa_mrope = is_function_constant_defined(QSA_MROPE) && QSA_MROPE;
constant bool QSA_ROWS_C [[function_constant(30)]];
constant bool qsa_rows = is_function_constant_defined(QSA_ROWS_C) && QSA_ROWS_C;
// QSA_ROWS (function constant 30): the rows of the window are DIFFERENT sequences (batch slots), one token each, each at its own
// cell and in its own slot of the caches.  A buffer rinfo (uint) carries the strides between the slots' caches, in elements:
// [0] kq / vq (chars), [1] ksc / vsc (halves), [2] tail, [3] dead, [4] pooled (floats), [5] GDN state (floats), [6] conv history
// (floats, per slot and parity), [7] PLE history (floats); then row t at [8 + 4 t] = {slot, cell, history parity, 0}.  Off: exactly the
// plain kernels.

struct QsaArgs {
    uint pos0;        // the window's first cell (relative to the sequence)
    uint pos_base;    // absolute position of cell 0 (RoPE)
    uint max_cells;
    uint n_tok;
    float eps;
};

// cache layout per layer
//   kq/vq [cell][kv head][256] int8, ks/vs [cell][kv head][4] half
//   tail [4][128] f32 (slot = cell % 4), dead [128], pooled [max_cells / 4 + 1][128]

static inline float tg_sum256(float v, threadgroup float* part, ushort sg, ushort lane, ushort nsg) {
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (int i = 0; i < nsg; ++i) t += part[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// one row (width w <= 256 = threads) : out = rope((rs * x) * g)  (RoPE on the first 64 dims, NEOX pairs i, i+32)
static inline float norm_rope(float x, float xhi_unused, device const float* g, threadgroup float* row,
                              threadgroup float* part, ushort tid, ushort sg, ushort lane, int w, float eps,
                              device const float2* cs, uint absp, bool rope, device const int* mtab) {
    const float ss = tg_sum256(tid < w ? x * x : 0.0f, part, sg, lane, 8);
    const float rs = rsqrt(ss / float(w) + eps);
    const float y = tid < w ? (rs * x) * g[tid] : 0.0f;
    row[tid] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = y;
    if (rope && tid < QROT) {
        const int p = tid & 31;
        const uint pp = qsa_mrope ? uint(mtab[absp * 3 + uint(p % 3)]) : absp;
        const float2 c = cs[pp * 32 + p];
        const float a = row[p], b = row[p + 32];
        o = tid < 32 ? a * c.x - b * c.y : a * c.y + b * c.x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (void)xhi_unused;
    return o;
}

// grid (32 rows + 1, n_tok): rows 0..23 q heads, 24..25 k heads, 26..27 v heads, 28..31 indexer q heads;
// threadgroup 32 (x = 32, y = 0) is the indexer-key threadgroup.  256 threads.
kernel void qsa_prep(device const float* qfull [[buffer(0)]], device const float* kcur [[buffer(1)]],
                     device const float* vcur [[buffer(2)]], device const float* iqraw [[buffer(3)]],
                     device const float* ikraw [[buffer(4)]], device const float* qn [[buffer(5)]],
                     device const float* kn [[buffer(6)]], device const float* iqn [[buffer(7)]],
                     device const float* ikn [[buffer(8)]], device const float2* cs [[buffer(9)]],
                     device float* qcur [[buffer(10)]], device float* qidx [[buffer(11)]],
                     device char* kq [[buffer(12)]], device char* vq [[buffer(13)]], device half* ksc [[buffer(14)]],
                     device half* vsc [[buffer(15)]], device float* tail [[buffer(16)]], device float* dead [[buffer(17)]],
                     device float* pooled [[buffer(18)]], constant QsaArgs& a [[buffer(19)]],
                     device const int* mtab [[buffer(20)]],
                     device const uint* rinfo [[buffer(21), function_constant(qsa_rows)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    threadgroup float row[256];
    threadgroup float amax[8];
    const int r = int(tg.x);
    if (r == 32) {
        // the indexer keys, every window token in order (only threadgroup (32, 0) works)
        if (tg.y != 0) return;
        for (int t = 0; t < int(a.n_tok); ++t) {
            uint cell = a.pos0 + uint(t);
            device float* tail_t = tail; device float* dead_t = dead; device float* pooled_t = pooled;
            if (qsa_rows) {
                const ulong sl = rinfo[8 + 4 * t];
                cell = rinfo[8 + 4 * t + 1];
                tail_t = tail + sl * rinfo[2]; dead_t = dead + sl * rinfo[3]; pooled_t = pooled + sl * rinfo[4];
            }
            const float x = tid < ID ? ikraw[t * ID + tid] : 0.0f;
            if (tid < ID) tail_t[(cell & 3) * ID + tid] = x;
            threadgroup_barrier(mem_flags::mem_device);
            if (cell == 0) {
                const float o = norm_rope(x, 0.0f, ikn, row, part, tid, sg, lane, ID, a.eps, cs, 0, false, mtab);
                if (tid < ID) dead_t[tid] = o;
            }
            if ((cell & 3) == 3) {
                float m = 0.0f;
                if (tid < ID) m = (((tail_t[tid] + tail_t[ID + tid]) + tail_t[2 * ID + tid]) + tail_t[3 * ID + tid]) * 0.25f;
                const uint b = cell >> 2;
                const float o = norm_rope(m, 0.0f, ikn, row, part, tid, sg, lane, ID, a.eps, cs, a.pos_base + 4 * b, true, mtab);
                if (tid < ID) pooled_t[b * ID + tid] = o;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        return;
    }
    const int t = int(tg.y);
    const uint cell = qsa_rows ? rinfo[8 + 4 * t + 1] : a.pos0 + uint(t), absp = a.pos_base + cell;
    device char* kqr = kq; device char* vqr = vq; device half* kscr = ksc; device half* vscr = vsc;
    if (qsa_rows) {
        const ulong sl = rinfo[8 + 4 * t];
        kqr = kq + sl * rinfo[0]; vqr = vq + sl * rinfo[0]; kscr = ksc + sl * rinfo[1]; vscr = vsc + sl * rinfo[1];
    }
    if (r < QH) {
        const float x = qfull[(t * QH + r) * 2 * QD + tid];
        const float o = norm_rope(x, 0.0f, qn, row, part, tid, sg, lane, QD, a.eps, cs, absp, true, mtab);
        qcur[(t * QH + r) * QD + tid] = o;
    } else if (r < QH + 2 * QKV) {
        const bool isv = r >= QH + QKV;
        const int h = r - QH - (isv ? QKV : 0);
        float x = (isv ? vcur : kcur)[(t * QKV + h) * QD + tid];
        if (!isv) x = norm_rope(x, 0.0f, kn, row, part, tid, sg, lane, QD, a.eps, cs, absp, true, mtab);
        // int8 with an fp16 scale per 64 values: groups = simdgroup pairs (64 threads)
        float m = simd_max(fabs(x));
        if (lane == 0) amax[sg] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = tid >> 6;
        m = max(amax[2 * g], amax[2 * g + 1]);
        const half sh = half(m / 127.0f);
        const float sf = float(sh);
        int q = 0;
        if (sf > 0.0f) q = clamp(int(rint(x / sf)), -127, 127);
        const uint rowi = cell * QKV + uint(h);
        (isv ? vqr : kqr)[rowi * QD + tid] = char(q);
        if ((tid & 63) == 0) (isv ? vscr : kscr)[rowi * 4 + g] = sh;
    } else {
        const int h = r - QH - 2 * QKV;
        const float x = tid < ID ? iqraw[(t * IH + h) * ID + tid] : 0.0f;
        const float o = norm_rope(x, 0.0f, iqn, row, part, tid, sg, lane, ID, a.eps, cs, absp, true, mtab);
        if (tid < ID) qidx[(t * IH + h) * ID + tid] = o;
    }
}

// scores [n_tok][max_blocks]; grid (ceil(max_blocks / 8), n_tok), 256 threads: a simdgroup per block.
kernel void qsa_score(device const float* qidx [[buffer(0)]], device const float* pooled [[buffer(1)]],
                      device const float* dead [[buffer(2)]], device float* scores [[buffer(3)]],
                      constant QsaArgs& a [[buffer(4)]],
                      device const uint* rinfo [[buffer(5), function_constant(qsa_rows)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    const int t = int(tg.y);
    const uint n_kv = (qsa_rows ? rinfo[8 + 4 * t + 1] : a.pos0 + uint(t)) + 1, n_bid = n_kv >> 2;
    const uint b = tg.x * 8 + sg;
    if (b > n_bid) return;
    device const float* pooled_s = pooled; device const float* dead_s = dead;
    if (qsa_rows) { const ulong sl = rinfo[8 + 4 * t]; pooled_s = pooled + sl * rinfo[4]; dead_s = dead + sl * rinfo[3]; }
    device const float* key = b == n_bid ? dead_s : pooled_s + b * ID;
    float k[4];
    for (int i = 0; i < 4; ++i) k[i] = key[lane + 32 * i];
    float s = 0.0f;
    for (int h = 0; h < IH; ++h) {
        float d = 0.0f;
        for (int i = 0; i < 4; ++i) d = fma(qidx[(t * IH + h) * ID + lane + 32 * i], k[i], d);
        d = simd_sum(d);
        s += max(d, 0.0f);
    }
    if (b == n_bid && (n_kv & 3) != 0) s += 1e9f;
    if (lane == 0) scores[uint(t) * (a.max_cells / 4 + 2) + b] = s;
}

// ids [n_tok][QBUDGET], ascending.  One threadgroup of 1024 threads per token.  Order key of a float: larger score =
// larger key; a radix select (4 passes of 8 bits) over the blocks with weights = their cell counts finds the
// threshold key T and how many cells with key == T to take (the lowest cells first).
static inline uint okey(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
kernel void qsa_topk(device const float* scores [[buffer(0)]], device int* ids [[buffer(1)]],
                     constant QsaArgs& a [[buffer(2)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint sel[4];
    threadgroup uint pref[32];
    const int t = int(tg.x);
    const uint n_kv = a.pos0 + uint(t) + 1, n_bid = n_kv >> 2, nb = n_bid + 1;
    device int* out = ids + t * QBUDGET;
    device const float* sc = scores + uint(t) * (a.max_cells / 4 + 2);
    if (n_kv <= QBUDGET) {
        for (uint c = tid; c < n_kv; c += 1024) out[c] = int(c);
        return;
    }
    // radix select of the `need`-th largest cell key
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
        if (tid == 0) {
            uint acc = 0, d = 0;
            for (int v = 255; v >= 0; --v) {
                const uint h = atomic_load_explicit(&hist[v], memory_order_relaxed);
                if (acc + h >= need) { d = uint(v); break; }
                acc += h;
            }
            sel[0] = d;
            sel[1] = need - acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= sel[0] << shift;
        pmask |= 255u << shift;
        need = sel[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // T = prefix; take every cell with key > T and the first `need` cells (lowest index) with key == T, ascending.
    // Pass over blocks in order of index with a prefix scan of each block's selected cell count.
    const uint T = prefix;
    uint base = 0, eq_taken = 0;
    for (uint b0 = 0; b0 < nb; b0 += 1024) {
        const uint b = b0 + tid;
        uint w = 0, weq = 0;
        if (b < nb) {
            const uint k = okey(sc[b]);
            const uint cells = b < n_bid ? 4u : (n_kv & 3u);
            if (k > T) w = cells;
            else if (k == T) weq = cells;
        }
        // equal-key cells: the lowest first, `need` of them in total
        const uint eq_before = simd_prefix_exclusive_sum(weq);
        if (lane == 31) pref[sg] = eq_before + weq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint eq_sg = 0;
        for (int i = 0; i < int(sg); ++i) eq_sg += pref[i];
        uint eq_tot = 0;
        for (int i = 0; i < 32; ++i) eq_tot += pref[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint eq0 = eq_taken + eq_sg + eq_before;
        uint take_eq = 0;
        if (weq && eq0 < need) take_eq = min(weq, need - eq0);
        const uint mine = w + take_eq;
        const uint before = simd_prefix_exclusive_sum(mine);
        if (lane == 31) pref[sg] = before + mine;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint off = base;
        for (int i = 0; i < int(sg); ++i) off += pref[i];
        uint tot = 0;
        for (int i = 0; i < 32; ++i) tot += pref[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        off += before;
        for (uint c = 0; c < mine; ++c) out[off + c] = int(4 * b + c);
        base += tot;
        eq_taken += eq_tot;
    }
}

// partials: [n_tok][chunks][QH][QD + 2] (acc, then m, l).  grid (chunks, QKV, n_tok), 256 threads.
kernel void qsa_attn(device const float* qcur [[buffer(0)]], device const char* kq [[buffer(1)]],
                     device const char* vq [[buffer(2)]], device const half* ksc [[buffer(3)]],
                     device const half* vsc [[buffer(4)]], device const int* ids [[buffer(5)]],
                     device float* part [[buffer(6)]], constant QsaArgs& a [[buffer(7)]],
                     device const uint* rinfo [[buffer(8), function_constant(qsa_rows)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HPG = QH / QKV;   // 12 query heads per kv head
    threadgroup float qs[HPG][QD];
    threadgroup float s[HPG][QCH];
    threadgroup int cid[QCH];
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = (qsa_rows ? rinfo[8 + 4 * t + 1] : a.pos0 + uint(t)) + 1, width = min(n_kv, uint(QBUDGET));
    if (qsa_rows) { const ulong sl = rinfo[8 + 4 * t]; kq += sl * rinfo[0]; vq += sl * rinfo[0]; ksc += sl * rinfo[1]; vsc += sl * rinfo[1]; }
    const int c0 = chunk * QCH;
    const int nchunks = (QBUDGET + QCH - 1) / QCH;
    device float* pp = part + ((uint(t) * nchunks + chunk) * QH) * (QD + 2);
    if (uint(c0) >= width) {
        // an empty chunk: m = -inf, l = 0 (the merge skips it)
        for (int h = 0; h < HPG; ++h)
            if (tid == 0) { pp[(g * HPG + h) * (QD + 2) + QD] = -INFINITY; pp[(g * HPG + h) * (QD + 2) + QD + 1] = 0.0f; }
        return;
    }
    const int nc = min(QCH, int(width) - c0);
    for (int h = 0; h < HPG; ++h) qs[h][tid] = qcur[(t * QH + g * HPG + h) * QD + tid];
    if (tid < QCH) cid[tid] = tid < nc ? ids[t * QBUDGET + c0 + tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: 12 x 64 dots of 256; simdgroup sg takes cells sg, sg + 8, ...; lanes split the 256 dims (8 each)
    for (int c = sg; c < nc; c += 8) {
        const uint rowi = uint(cid[c]) * QKV + uint(g);
        device const char* kr = kq + rowi * QD;
        float kv[8];
        const float scl = float(ksc[rowi * 4 + (lane >> 3)]);
        for (int i = 0; i < 8; ++i) kv[i] = float(kr[lane * 8 + i]) * scl;
        for (int h = 0; h < HPG; ++h) {
            float d = 0.0f;
            for (int i = 0; i < 8; ++i) d = fma(qs[h][lane * 8 + i], kv[i], d);
            d = simd_sum(d);
            if (lane == 0) s[h][c] = d * 0.0625f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // per head: max, exp, sum (simdgroup h % 8 handles heads h, h + 8)
    threadgroup float mh[HPG], lh[HPG];
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
    // acc[h][j] = sum_c p[h][c] v[c][j]: thread = dim j
    float acc[HPG];
    for (int h = 0; h < HPG; ++h) acc[h] = 0.0f;
    for (int c = 0; c < nc; ++c) {
        const uint rowi = uint(cid[c]) * QKV + uint(g);
        const float v = float(vq[rowi * QD + tid]) * float(vsc[rowi * 4 + (tid >> 6)]);
        for (int h = 0; h < HPG; ++h) acc[h] = fma(s[h][c], v, acc[h]);
    }
    for (int h = 0; h < HPG; ++h) {
        device float* o = pp + (g * HPG + h) * (QD + 2);
        o[tid] = acc[h];
        if (tid == 0) { o[QD] = mh[h]; o[QD + 1] = lh[h]; }
    }
}

// grid (QH, n_tok), 256 threads: out = merge * sigmoid(gate), as prepared activations of n = QH * QD per token
kernel void qsa_merge(device const float* part [[buffer(0)]], device const float* qfull [[buffer(1)]],
                      device float* xp [[buffer(2)]], constant QsaArgs& a [[buffer(3)]],
                      device const uint* rinfo [[buffer(4), function_constant(qsa_rows)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float msum[256];
    const int h = int(tg.x), t = int(tg.y);
    const int nchunks = (QBUDGET + QCH - 1) / QCH;
    const uint n_kv = (qsa_rows ? rinfo[8 + 4 * t + 1] : a.pos0 + uint(t)) + 1, width = min(n_kv, uint(QBUDGET));
    const int nc = int((width + QCH - 1) / QCH);
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
    // prepared planes (xprep layout) of the n = QH * QD vector
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
