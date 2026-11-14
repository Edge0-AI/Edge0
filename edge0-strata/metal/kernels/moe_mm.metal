// metal/kernels/moe_mm.metal - routed experts for a PROMPT chunk (after quant/units/mma.metal): a grouped GEMM on
// the matrix units through Metal 4's MetalPerformancePrimitives matmul2d (on the M5 its neural accelerators).
//
// Plan (deterministic, three passes):  the chunk's T x topk (token, expert) entries are sorted by expert, then by
// entry index; each expert's run is padded to a multiple of TM rows; tile i = (expert, first padded row).
//   mm_hist   per block of 1024 entries: a histogram of experts      -> bcnt[block][expert]
//   mm_scan   one threadgroup (thread = expert): block offsets, the padded expert offsets, the tile list
//   mm_place  per block: an entry's padded row = expert offset + earlier blocks' count + rank in its block;
//             srt_tok[row] = token (-1 on padding), srt_w[row] = weight, pos[entry] = row
// GEMMs: a threadgroup takes a tile (TM rows of one expert) and TN output rows; per K chunk of TK it converts the
// tile's activations to fp16 and dequantizes TN x TK weights (ref_quant's arithmetic, then rounded to fp16) into
// threadgroup memory, and matmul2d multiplies them, fp32 accumulation.
//   mm_gu     H[row][f] = silu(x . gate_f) * (x . up_f), fp32, for the n_ff rows
//   mm_down   YE[row][d] = h . down_d, fp32
//   mm_combine y[t][d] = sum_k w(t, k) * YE[pos(t, k)][d], k in order
//
// The idea of feeding matmul2d tiles of dequantized weights from threadgroup memory follows yibie/strata-mlx's
// tile_matmul (MIT); the code is our own.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmPlanArgs {
    uint n_ent;      // T * topk
    uint topk;
    uint n_expert;
    uint tm;         // the tile height (rows of one expert)
    uint n_blocks;   // ceil(n_ent / 1024)
};

// counts per block: bcnt[block * n_expert + e]
kernel void mm_hist(device const int* ids [[buffer(0)]], device uint* bcnt [[buffer(1)]],
                    constant MmPlanArgs& a [[buffer(2)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint h[1024];
    for (uint e = tid; e < a.n_expert; e += 1024) atomic_store_explicit(&h[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = tg.x * 1024 + tid;
    if (i < a.n_ent) atomic_fetch_add_explicit(&h[ids[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < a.n_expert; e += 1024)
        bcnt[tg.x * a.n_expert + e] = atomic_load_explicit(&h[e], memory_order_relaxed);
}

// one threadgroup of n_expert (<= 1024) threads.  boff[block][e] = expert e's rows before this block (within e);
// eoff[e] = expert e's first padded row; tiles[i] = (expert, row0); meta[0] = n_tiles, meta[1] = padded rows;
// indirect grids: meta[2..4] = (gx_gu, n_tiles, 1) for the gate/up GEMM, meta[5..7] = (gx_down, n_tiles, 1) for the
// down GEMM (gx_gu in meta[2], gx_down in meta[5] written by the host once), meta[8..10] = (rows, 1, 1) for mm_gather.
kernel void mm_scan(device const uint* bcnt [[buffer(0)]], device uint* boff [[buffer(1)]],
                    device uint* eoff [[buffer(2)]], device int2* tiles [[buffer(3)]], device uint* meta [[buffer(4)]],
                    constant MmPlanArgs& a [[buffer(5)]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint s_rows[32], s_tiles[32];
    // simdgroups past the launched ones must count 0 (512 threads fill only 16 of the 32 slots)
    if (tid < 32) { s_rows[tid] = 0; s_tiles[tid] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint e = tid;
    uint tot = 0;
    if (e < a.n_expert)
        for (uint b = 0; b < a.n_blocks; ++b) {
            boff[b * a.n_expert + e] = tot;
            tot += bcnt[b * a.n_expert + e];
        }
    const uint nt = (tot + a.tm - 1) / a.tm, rows = nt * a.tm;
    const uint pr = simd_prefix_exclusive_sum(rows), pt = simd_prefix_exclusive_sum(nt);
    if (lane == 31) { s_rows[sg] = pr + rows; s_tiles[sg] = pt + nt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint orow = 0, otile = 0, all_rows = 0, all_tiles = 0;
    for (int i = 0; i < 32; ++i) {
        if (i < int(sg)) { orow += s_rows[i]; otile += s_tiles[i]; }
        all_rows += s_rows[i];
        all_tiles += s_tiles[i];
    }
    if (e < a.n_expert) {
        const uint r0 = orow + pr, t0 = otile + pt;
        eoff[e] = r0;
        for (uint k = 0; k < nt; ++k) tiles[t0 + k] = int2(int(e), int(r0 + k * a.tm));
    }
    if (tid == 0) {
        meta[0] = all_tiles;
        meta[1] = all_rows;
        meta[3] = all_tiles;   // indirect grid of the gate/up GEMM: (meta[2] given, tiles, 1)
        meta[4] = 1;
        meta[6] = all_tiles;   // of the down GEMM: (meta[5] given, tiles, 1)
        meta[7] = 1;
        meta[8] = all_rows;    // of the gather: (rows, 1, 1)
        meta[9] = 1;
        meta[10] = 1;
    }
}

// grid (n_blocks), 1024 threads.  srt_* must be cleared to the padding value (-1 / 0) beforehand by mm_clear.
kernel void mm_place(device const int* ids [[buffer(0)]], device const float* w [[buffer(1)]],
                     device const uint* boff [[buffer(2)]], device const uint* eoff [[buffer(3)]],
                     device int* srt_tok [[buffer(4)]], device float* srt_w [[buffer(5)]], device int* pos [[buffer(6)]],
                     constant MmPlanArgs& a [[buffer(7)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup int s_id[1024];
    const uint i = tg.x * 1024 + tid;
    const int e = i < a.n_ent ? ids[i] : -1;
    s_id[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= a.n_ent) return;
    uint rank = 0;
    for (uint j = 0; j < tid; ++j) rank += s_id[j] == e;
    const uint row = eoff[e] + boff[tg.x * a.n_expert + e] + rank;
    srt_tok[row] = int(i / a.topk);
    srt_w[row] = w[i];
    pos[i] = int(row);
}

kernel void mm_clear(device int* srt_tok [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) srt_tok[gid] = -1;
}

// ---- the GEMMs -----------------------------------------------------------------------------------------------------
// weights of one expert: gate at W + gate_off + e * gu_stride (+ up_off for up), down at W + down_off + e * d_stride
struct MmArgs {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong gu_stride;
    ulong d_stride;
    uint gu_row;     // bytes per gate/up row (n_embd values)
    uint d_row;      // bytes per down row (n_ff values)
    uint n_embd;
    uint n_ff;
};

// TN x TK weights of `nrow`-strided rows into threadgroup half: thread-strided units of the format
template <int F, int TN, int TK>
static inline void stage_w(device const uchar* w0, uint row_bytes, int k0, threadgroup half* dst,
                           threadgroup const uint* tab, ushort tid, ushort nth) {
    constexpr int UNIT = fmt<F>::UNIT, UPC = TK / UNIT, H = UNIT / 2;
    static_assert(TK % UNIT == 0, "the K chunk must hold whole units of the format");
    for (int it = tid; it < TN * UPC; it += nth) {
        const int r = it / UPC, uu = it - r * UPC;
        float wv[UNIT];
        int p0, p1;
        DW<F>::run(w0 + ulong(r) * row_bytes, k0 / UNIT + uu, k0, wv, p0, p1, tab);
        for (int j = 0; j < H; ++j) {
            dst[r * TK + p0 + j] = half(wv[j]);
            dst[r * TK + p1 + j] = half(wv[H + j]);
        }
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_gu(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                  device float* H [[buffer(2)]], device const int* srt_tok [[buffer(3)]],
                  device const int2* tiles [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                  uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half xa[TM * TK];
    threadgroup half wg[TN * TK];
    threadgroup half wu[TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup int tok[TM];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, f0 = int(tg.x) * TN;
    for (int m = tid; m < TM; m += NTH) tok[m] = srt_tok[row0 + m];
    device const uchar* g0 = W + a.gate_off + ulong(e) * a.gu_stride + ulong(f0) * a.gu_row;
    device const uchar* u0 = W + a.up_off + ulong(e) * a.gu_stride + ulong(f0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(xa, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tG(wg, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tU(wu, dextents<int32_t, 2>(TK, TN));
    auto accg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    auto accu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        for (int it = tid; it < TM * TK; it += NTH) {
            const int m = it / TK, k = it - m * TK;
            xa[it] = tok[m] >= 0 ? half(x[ulong(tok[m]) * a.n_embd + k0 + k]) : half(0.0f);
        }
        stage_w<F, TN, TK>(g0, a.gu_row, k0, wg, tab, tid, NTH);
        stage_w<F, TN, TK>(u0, a.gu_row, k0, wu, tab, tid, NTH);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto pg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
        auto pu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
        op.run(tA, tG, pg);
        op.run(tA, tU, pu);
        for (uint16_t i = 0; i < pg.get_capacity(); ++i)
            if (pg.is_valid_element(i)) { accg[i] += pg[i]; accu[i] += pu[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = accg[i];
            accg[i] = g / (1.0f + exp(-g)) * accu[i];
        }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tH(H, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    accg.store(tH.slice(f0, row0));
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_down(device const uchar* W [[buffer(0)]], device const float* H [[buffer(1)]],
                    device float* YE [[buffer(2)]], device const int2* tiles [[buffer(3)]],
                    constant MmArgs& a [[buffer(4)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half ha[TM * TK];
    threadgroup half wd[TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, d0 = int(tg.x) * TN;
    device const uchar* w0 = W + a.down_off + ulong(e) * a.d_stride + ulong(d0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(ha, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(wd, dextents<int32_t, 2>(TK, TN));
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        for (int it = tid; it < TM * TK; it += NTH) {
            const int m = it / TK, k = it - m * TK;
            ha[it] = half(H[ulong(row0 + m) * a.n_ff + k0 + k]);
        }
        stage_w<F, TN, TK>(w0, a.d_row, k0, wd, tab, tid, NTH);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto p = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
        op.run(tA, tW, p);
        for (uint16_t i = 0; i < p.get_capacity(); ++i)
            if (p.is_valid_element(i)) acc[i] += p[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    acc.store(tY.slice(d0, row0));
}

// grid (T * n_embd / 256), 256 threads
kernel void mm_combine(device const float* YE [[buffer(0)]], device const int* pos [[buffer(1)]],
                       device const float* srt_w [[buffer(2)]], device float* y [[buffer(3)]],
                       constant uint4& a [[buffer(4)]],   // (T, n_embd, topk, -)
                       uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.y, d = gid % a.y;
    if (t >= a.x) return;
    float s = 0.0f;
    for (uint k = 0; k < a.z; ++k) {
        const int r = pos[t * a.z + k];
        s = fma(srt_w[r], YE[ulong(r) * a.y + d], s);
    }
    y[gid] = s;
}

#define MMGU_(F, M, N, K, S) template [[host_name("mm_gu_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_gu<F, M, N, K, S>) mm_gu<F, M, N, K, S>;
#define MMDN_(F, M, N, K, S) template [[host_name("mm_down_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_down<F, M, N, K, S>) mm_down<F, M, N, K, S>;
#define MMGU(F, M, N, K, S) MMGU_(F, M, N, K, S)
#define MMDN(F, M, N, K, S) MMDN_(F, M, N, K, S)
#define MM_GU_SET(F) MMGU(F, 32, 64, 64, 4) MMGU(F, 64, 64, 64, 4) MMGU(F, 16, 64, 64, 2) MMGU(F, 32, 32, 64, 2)
#define MM_DN_SET(F) MMDN(F, 32, 64, 64, 4) MMDN(F, 64, 64, 64, 4) MMDN(F, 16, 64, 64, 2) MMDN(F, 32, 32, 64, 2)
MM_GU_SET(T_Q2_0) MM_GU_SET(T_IQ3_XXS) MM_GU_SET(T_IQ2_XXS) MM_GU_SET(T_IQ2_XS) MM_GU_SET(T_IQ2_S) MM_GU_SET(T_IQ3_S)
MM_DN_SET(T_Q2_0) MM_DN_SET(T_IQ4_NL)

// ---- v2: activations gathered once per layer as fp16 rows, read by matmul2d from device memory ---------------------
// mm_gather: XS[row][k] = half(x[srt_tok[row]][k]) (0 on padding rows).  grid (rows), 256 threads.
kernel void mm_gather(device const float* x [[buffer(0)]], device const int* srt_tok [[buffer(1)]],
                      device half* XS [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const int t = srt_tok[tg.x];
    device half* o = XS + ulong(tg.x) * n;
    for (uint k = tid; k < n; k += 256) o[k] = t >= 0 ? half(x[ulong(t) * n + k]) : half(0.0f);
}

// The SSD-streaming arena (as in moe.metal): with function constant 20, expert e's slot is smap[e] in the chunks W,
// W1..W3 (buffers 5..8 of mm_gu2 / mm_down2 / mm_gu3 / mm_down3); gate_off / up_off / down_off are offsets in the slot.
#ifndef STRATA_ARENA_METAL
#define STRATA_ARENA_METAL
constant bool kArenaFc [[function_constant(20)]];
constant bool kArena = is_function_constant_defined(kArenaFc) && kArenaFc;
static inline device const uchar* arena_slot(uint v, device const uchar* W, device const uchar* W1, device const uchar* W2,
                                             device const uchar* W3) {
    const uint c = v >> 28;
    device const uchar* b = c == 0 ? W : (c == 1 ? W1 : (c == 2 ? W2 : W3));
    return b + (ulong)(v & 0x0FFFFFFFu) * 16384ul;
}
#endif

// a probe for the bench (function constant 5): 1 = the weights not decoded (a cheap fill), 2 = no matrix multiply
constant int MM_PROBE [[function_constant(5)]];
constant int mm_probe = is_function_constant_defined(MM_PROBE) ? MM_PROBE : 0;

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_gu2(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]],
                   device half* HS [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]], device const uint* smap [[buffer(5)]],
                     device const uchar* W1 [[buffer(6)]], device const uchar* W2 [[buffer(7)]], device const uchar* W3 [[buffer(8)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half wg[TN * TK];
    threadgroup half wu[TN * TK];
    threadgroup float ot[TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, f0 = int(tg.x) * TN;
    device const uchar* E = kArena ? arena_slot(smap[e], W, W1, W2, W3) : W + ulong(e) * a.gu_stride;
    device const uchar* g0 = E + a.gate_off + ulong(f0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(f0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tG(wg, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tU(wu, dextents<int32_t, 2>(TK, TN));
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    auto accu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        if (mm_probe == 1) {
            for (int it = tid; it < TN * TK; it += NTH) { wg[it] = half(k0 & 7); wu[it] = half(it & 7); }
        } else {
            stage_w<F, TN, TK>(g0, a.gu_row, k0, wg, tab, tid, NTH);
            stage_w<F, TN, TK>(u0, a.gu_row, k0, wu, tab, tid, NTH);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = tX.template slice<TK, TM>(k0, row0);
        if (mm_probe != 2) {
            op.run(tA, tG, accg);
            op.run(tA, tU, accu);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = accg[i];
            accg[i] = g / (1.0f + exp(-g)) * accu[i];
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = tid; it < TM * TN; it += NTH) {
        const int m = it / TN, f = it - m * TN;
        HS[ulong(row0 + m) * a.n_ff + f0 + f] = half(ot[it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_down2(device const uchar* W [[buffer(0)]], device const half* HS [[buffer(1)]],
                     device float* YE [[buffer(2)]], device const int2* tiles [[buffer(3)]],
                     constant MmArgs& a [[buffer(4)]], device const uint* smap [[buffer(5)]],
                     device const uchar* W1 [[buffer(6)]], device const uchar* W2 [[buffer(7)]], device const uchar* W3 [[buffer(8)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half wd[TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, d0 = int(tg.x) * TN;
    device const uchar* w0 = (kArena ? arena_slot(smap[e], W, W1, W2, W3) : W + ulong(e) * a.d_stride) + a.down_off + ulong(d0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(wd, dextents<int32_t, 2>(TK, TN));
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        stage_w<F, TN, TK>(w0, a.d_row, k0, wd, tab, tid, NTH);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, tW, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    acc.store(tY.slice(d0, row0));
}

#define MMGU2_(F, M, N, K, S) template [[host_name("mm_gu2_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_gu2<F, M, N, K, S>) mm_gu2<F, M, N, K, S>;
#define MMDN2_(F, M, N, K, S) template [[host_name("mm_down2_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_down2<F, M, N, K, S>) mm_down2<F, M, N, K, S>;
#define MMGU2(F, M, N, K, S) MMGU2_(F, M, N, K, S)
#define MMDN2(F, M, N, K, S) MMDN2_(F, M, N, K, S)
#define MM_GU2_SET(F) MMGU2(F, 32, 64, 32, 2) MMGU2(F, 32, 32, 32, 2) MMGU2(F, 32, 64, 16, 2) MMGU2(F, 64, 32, 16, 2) MMGU2(F, 16, 32, 32, 2) MMGU2(F, 16, 64, 16, 2) MMGU2(F, 16, 128, 32, 4) MMGU2(F, 32, 64, 16, 4) MMGU2(F, 16, 32, 32, 1) MMGU2(F, 16, 64, 32, 4) MMGU2(F, 16, 64, 64, 2) MMGU2(F, 32, 64, 64, 4) MMGU2(F, 64, 64, 64, 4) MMGU2(F, 16, 32, 64, 2) MMGU2(F, 32, 64, 32, 4) MMGU2(F, 64, 64, 32, 4) MMGU2(F, 16, 64, 32, 2)
#define MM_DN2_SET(F) MMDN2(F, 32, 64, 32, 2) MMDN2(F, 32, 32, 32, 2) MMDN2(F, 32, 64, 16, 2) MMDN2(F, 64, 32, 16, 2) MMDN2(F, 16, 32, 32, 2) MMDN2(F, 16, 64, 16, 2) MMDN2(F, 16, 128, 32, 4) MMDN2(F, 32, 64, 16, 4) MMDN2(F, 16, 32, 32, 1) MMDN2(F, 16, 64, 32, 4) MMDN2(F, 16, 64, 64, 2) MMDN2(F, 32, 64, 64, 4) MMDN2(F, 64, 64, 64, 4) MMDN2(F, 16, 32, 64, 2) MMDN2(F, 32, 64, 32, 4) MMDN2(F, 64, 64, 32, 4) MMDN2(F, 16, 64, 32, 2)
#define MM_GU2_U32(F) MMGU2(F, 16, 64, 32, 2) MMGU2(F, 16, 64, 64, 2) MMGU2(F, 32, 64, 32, 2) MMGU2(F, 16, 32, 32, 2) \
    MMGU2(F, 64, 64, 32, 4)
#define MM_DN2_U32(F) MMDN2(F, 16, 64, 32, 2) MMDN2(F, 16, 64, 64, 2) MMDN2(F, 16, 32, 32, 2) MMDN2(F, 32, 32, 32, 2) \
    MMDN2(F, 16, 128, 32, 4) MMDN2(F, 32, 64, 32, 2) MMDN2(F, 64, 64, 32, 4)
// K-quants: the staged K chunk must hold whole units as DW places them (128 wide): TK 128
#define MM_GU2_KQ(F) MMGU2(F, 16, 32, 128, 2) MMGU2(F, 16, 16, 128, 2) MMGU2(F, 32, 32, 128, 2) MMGU2(F, 16, 32, 128, 4)
MM_GU2_KQ(T_Q4_K) MM_GU2_KQ(T_Q5_K) MM_GU2_KQ(T_Q6_K) MM_GU2_U32(T_Q8_0) MM_DN2_U32(T_Q5_1) MM_DN2_U32(T_Q8_0)
// IQ4_XS (UNIT 32): the K chunk must be >= 32
MMGU2(T_IQ4_XS, 16, 64, 32, 2) MMGU2(T_IQ4_XS, 16, 64, 64, 2) MMGU2(T_IQ4_XS, 32, 64, 32, 2) MMGU2(T_IQ4_XS, 16, 32, 32, 2)
MMGU2(T_IQ4_XS, 64, 64, 32, 4)
MM_GU2_SET(T_Q2_0) MM_GU2_SET(T_IQ3_XXS) MM_GU2_SET(T_IQ2_XXS) MM_GU2_SET(T_IQ2_XS) MM_GU2_SET(T_IQ2_S) MM_GU2_SET(T_IQ3_S)
MMGU2(T_Q3_K, 16, 64, 64, 2)
MMGU2(T_Q3_K, 32, 64, 32, 2)
MMDN2(T_IQ4_NL, 32, 64, 32, 2)
MM_GU2_SET(T_TQK6) MM_GU2_SET(T_TQ2T)   // Gyro: the trellis gate/up (units of 16 weights: TK 16 or a multiple)
MM_DN2_SET(T_Q2_0)
MM_DN2_SET(T_TQK7) MM_DN2_SET(T_TQ2T)
MMDN2(T_IQ4_NL, 16, 64, 32, 2) MMDN2(T_IQ4_NL, 16, 64, 64, 2) MMDN2(T_IQ4_NL, 16, 32, 32, 2) MMDN2(T_IQ4_NL, 32, 32, 32, 2) MMDN2(T_IQ4_NL, 16, 128, 32, 4)

// ---- v3: the weights decoded straight into matmul2d's right-input cooperative tensor (single-simdgroup scope),
// no threadgroup staging and no barriers in the K loop.  Each simdgroup owns TN output rows; a threadgroup SG * TN.
// The right input's element layout (tests/coop_layout): runs of 4 consecutive k at one n, so a lane decodes 4 values
// at a time (W4<F>).  Activations: the gathered fp16 rows from device memory.
#ifndef STRATA_COOP_K4
#define STRATA_COOP_K4
// function constant 7: the right-input cooperative tensor holds runs of 4 consecutive k at one n per lane (the M5's
// layout; tests/coop_layout).  false (the M4: runs of 2 n at one k): every element decoded on its own.
constant bool COOP_K4 [[function_constant(7)]];
constant bool coop_k4 = is_function_constant_defined(COOP_K4) ? COOP_K4 : true;
#endif
template <int F> struct W4;
// the four weights k .. k + 3 (k % 4 == 0) of a row, in quant.metal's deq arithmetic (code * scale in fp32), as fp16
// the threadgroup forms of quant.metal's field readers (blocks staged in threadgroup memory)
static inline uint rd32(threadgroup const uchar* p) {
    threadgroup const ushort* s = (threadgroup const ushort*)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
static inline float rdh(threadgroup const uchar* p) { return float(*(threadgroup const half*)p); }

static inline half4 w4_signed(uint g, uint s, float sc) {
    const uchar4 v = as_type<uchar4>(g);
    const float4 q = float4((s & 1) ? -float(v[0]) : float(v[0]), (s & 2) ? -float(v[1]) : float(v[1]),
                            (s & 4) ? -float(v[2]) : float(v[2]), (s & 8) ? -float(v[3]) : float(v[3]));
    return half4(q * sc);
}
template <> struct W4<T_Q2_0> {   // 64 values a block: fp16 d, 16 bytes of 2-bit codes (value v at byte v / 4)
    // fp32 math then fp16 (an fp16-math decode was measured slower: 4.2 vs 3.5 ms a layer)
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 6) * 18;
        const float d = float(*(device const half*)b);
        const uint q = b[2 + ((k & 63) >> 2)];
        return half4(float4(float(int(q & 3) - 1), float(int((q >> 2) & 3) - 1), float(int((q >> 4) & 3) - 1),
                            float(int(q >> 6) - 1)) * d);
    }
};
template <> struct W4<T_IQ3_XXS> {
    static inline half4 run(device const uchar* row, int k, threadgroup const uint* tab) {
        device const uchar* b = row + (k >> 8) * 98;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint aux = rd32(b + 2 + 64 + 4 * ib);
        threadgroup const uchar* ks = (threadgroup const uchar*)(tab + 256);
        const uint s = ks[(aux >> (7 * l)) & 127];
        const float sc = rdh(b) * (0.5f + float(aux >> 28)) * 0.5f;
        return w4_signed(tab[b[2 + 8 * ib + 2 * l + hh]], hh ? s >> 4 : s, sc);
    }
};
template <> struct W4<T_IQ2_XXS> {
    static inline half4 run(device const uchar* row, int k, threadgroup const uint* tab) {
        device const uchar* b = row + (k >> 8) * 66;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint a0 = rd32(b + 2 + 8 * ib), a1 = rd32(b + 2 + 8 * ib + 4);
        threadgroup const uchar* ks = (threadgroup const uchar*)(tab + 512);
        const uint idx = (a0 >> (8 * l)) & 255, s = ks[(a1 >> (7 * l)) & 127];
        const float sc = rdh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
        return w4_signed(tab[2 * idx + hh], hh ? s >> 4 : s, sc);
    }
};
template <> struct W4<T_IQ2_XS> {
    static inline half4 run(device const uchar* row, int k, threadgroup const uint* tab) {
        device const uchar* b = row + (k >> 8) * 74;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint v = ((device const ushort*)(b + 2))[4 * ib + l];
        threadgroup const uchar* ks = (threadgroup const uchar*)(tab + 1024);
        const uint idx = v & 511, s = ks[v >> 9];
        const uint scb = b[66 + ib];
        const float sc = rdh(b) * (0.5f + float((scb >> (4 * (l >> 1))) & 15)) * 0.25f;
        return w4_signed(tab[2 * idx + hh], hh ? s >> 4 : s, sc);
    }
};
template <> struct W4<T_IQ2_S> {
    static inline half4 run(device const uchar* row, int k, threadgroup const uint* tab) {
        device const uchar* b = row + (k >> 8) * 82;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint qh = b[66 + ib];
        const uint idx = uint(b[2 + 4 * ib + l]) | ((qh << (8 - 2 * l)) & 0x300);
        const uint s = b[2 + 32 + 4 * ib + l];
        const uint scb = b[74 + ib];
        const float sc = rdh(b) * (0.5f + float((scb >> (4 * (l >> 1))) & 15)) * 0.25f;
        return w4_signed(tab[2 * idx + hh], hh ? s >> 4 : s, sc);
    }
};
template <> struct W4<T_IQ3_S> {
    static inline half4 run(device const uchar* row, int k, threadgroup const uint* tab) {
        device const uchar* b = row + (k >> 8) * 110;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint qh = b[66 + ib];
        const uint i = hh ? (uint(b[2 + 8 * ib + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256))
                          : (uint(b[2 + 8 * ib + 2 * l]) | ((qh << (8 - 2 * l)) & 256));
        const uint s = b[74 + 4 * ib + l];
        const uint scb = b[106 + (ib >> 1)];
        const float sc = rdh(b) * float(1 + 2 * ((scb >> (4 * (ib & 1))) & 15));
        return w4_signed(tab[i], hh ? s >> 4 : s, sc);
    }
};
template <> struct W4<T_IQ4_XS> {   // 256 a block: d, scales_h, scales_l[4], qs[128]; 32-groups as IQ4_NL's
    static inline half4 run(device const uchar* row, int k, threadgroup const uint* tab) {
        device const uchar* b = row + (k >> 8) * 136;
        const int j = k & 255, ib = j >> 5, o = j & 15, hi = (j >> 4) & 1;
        const uint sh = *(device const ushort*)(b + 2);
        const int ls = int((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | int(((sh >> (2 * ib)) & 3) << 4);
        const float s = rdh(b) * float(ls - 32);
        device const uchar* qs = b + 8 + 16 * ib;
        float4 q;
        for (int c = 0; c < 4; ++c) {
            const half2 w = as_type<half2>(tab[qs[o + c]]);
            q[c] = float(hi ? w.y : w.x);
        }
        return half4(q * s);
    }
};
template <> struct W4<T_IQ4_NL> {   // 32 a block: d, 16 bytes; low nibbles are elements 0..15, high 16..31
    static inline half4 run(device const uchar* row, int k, threadgroup const uint* tab) {
        device const uchar* b = row + (k >> 5) * 18;
        const int j = k & 31, hi = j >> 4, o = j & 15;
        const float d = rdh(b);
        float4 q;
        for (int c = 0; c < 4; ++c) {
            const half2 w = as_type<half2>(tab[b[2 + o + c]]);
            q[c] = float(hi ? w.y : w.x);
        }
        return half4(q * d);
    }
};

// W8<F>: the runs k .. k + 3 and k + 16 .. k + 19 of a row (k % 32 < 16): one 32-group, so the per-group fields (scale,
// sign word) are read once.  The generic one is two W4.
// ---- the K-quant / legacy-quant W4 decoders (the UD-Q4/5/6_K_XL packs' experts): values k .. k + 3 (k % 4 == 0)
// of a row in ref_quant's arithmetic (float), then fp16
template <> struct W4<T_Q8_0> {   // 32 a block: d, int8 qs[32]
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 5) * 34;
        const float d = rdh(b);
        device const char* q = (device const char*)(b + 2 + (k & 31));
        return half4(float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) * d);
    }
};
template <> struct W4<T_Q5_1> {   // 32 a block: d, m, qh (32 bits), qs[16] (low nibbles 0..15, high 16..31)
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 5) * 24;
        const float d = rdh(b), m = rdh(b + 2);
        const uint hb = rd32(b + 4);
        const int j = k & 31, o = j & 15, hi = j >> 4;
        float4 w;
        for (int c = 0; c < 4; ++c) {
            const uint qv = b[8 + o + c];
            const int x = int(hi ? (qv >> 4) : (qv & 15)) | int(((hb >> (j + c)) & 1) << 4);
            w[c] = float(x) * d + m;
        }
        return half4(w);
    }
};
template <> struct W4<T_Q4_K> {   // 256 a block: d, dmin, scales[12], qs[128] (64-chunks: low nibbles then high)
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 8) * 144;
        const int j = k & 255, c = j >> 6, hi = (j >> 5) & 1, l = j & 31;
        float s, mm;
        scale_min_k4(2 * c + hi, b + 4, s, mm);
        const float d1 = rdh(b) * s, m1 = rdh(b + 2) * mm;
        device const uchar* q = b + 16 + 32 * c + l;
        float4 w;
        for (int i = 0; i < 4; ++i) w[i] = d1 * float(hi ? (q[i] >> 4) : (q[i] & 15)) - m1;
        return half4(w);
    }
};
template <> struct W4<T_Q5_K> {   // 256 a block: d, dmin, scales[12], qh[32], qs[128]
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 8) * 176;
        const int j = k & 255, c = j >> 6, hi = (j >> 5) & 1, l = j & 31;
        float s, mm;
        scale_min_k4(2 * c + hi, b + 4, s, mm);
        const float d1 = rdh(b) * s, m1 = rdh(b + 2) * mm;
        device const uchar* qh = b + 16 + l;
        device const uchar* ql = b + 48 + 32 * c + l;
        const uint bit = 1u << (2 * c + hi);
        float4 w;
        for (int i = 0; i < 4; ++i)
            w[i] = d1 * float(int(hi ? (ql[i] >> 4) : (ql[i] & 15)) + ((qh[i] & bit) ? 16 : 0)) - m1;
        return half4(w);
    }
};
template <> struct W4<T_Q6_K> {   // 256 a block: ql[128], qh[64], int8 scales[16], d; halves of 128
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 8) * 210;
        const int j = k & 255, n = j >> 7, jj = j & 127, qd = jj >> 5, l = jj & 31;
        device const uchar* ql = b + 64 * n + l + ((qd & 1) ? 32 : 0);
        device const uchar* qh = b + 128 + 32 * n + l;
        const float d = rdh(b + 208);
        const float ds = d * float(((device const char*)(b + 192 + 8 * n))[l / 16 + 2 * qd]);
        float4 w;
        for (int i = 0; i < 4; ++i) {
            const int q = int(((qd >> 1) ? (ql[i] >> 4) : (ql[i] & 15)) | (((qh[i] >> (2 * qd)) & 3) << 4)) - 32;
            w[i] = ds * float(q);
        }
        return half4(w);
    }
};

template <int F> struct W8 {
    static inline void run(device const uchar* row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        a = W4<F>::run(row, k, tab);
        b = W4<F>::run(row, k + 16, tab);
    }
};
// (a W8<T_Q2_0> sharing the block scale was measured slower than the two W4 of the generic one: 5.5 vs 3.5-4.2 ms)
template <> struct W8<T_IQ3_XXS> {
    template <typename P>
    static inline void run(P row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        const P bl = row + (k >> 8) * 98;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint aux = rd32(bl + 2 + 64 + 4 * ib);
        threadgroup const uchar* ks = (threadgroup const uchar*)(tab + 256);
        const float sc = rdh(bl) * (0.5f + float(aux >> 28)) * 0.5f;
        const P qs = bl + 2 + 8 * ib + hh;
        if (mm_probe == 3) {   // bench probe: no threadgroup-table lookups (wrong values)
            a = w4_signed(qs[2 * l] * 0x01010101u, aux, sc);
            b = w4_signed(qs[2 * l + 4] * 0x01010101u, aux >> 7, sc);
            return;
        }
        uint s0, s1;
        if (mm_probe == 4) {   // the signs computed (ksigns_iq2xs[i] = i | parity(i) << 7) instead of looked up
            const uint i0 = (aux >> (7 * l)) & 127, i1 = (aux >> (7 * (l + 2))) & 127;
            s0 = i0 | ((popcount(i0) & 1u) << 7);
            s1 = i1 | ((popcount(i1) & 1u) << 7);
        } else {
            s0 = ksg(tab, 256, (aux >> (7 * l)) & 127);
            s1 = ksg(tab, 256, (aux >> (7 * (l + 2))) & 127);
        }
        a = w4_signed(g_iq3xxs(tab, qs[2 * l]), hh ? s0 >> 4 : s0, sc);
        b = w4_signed(g_iq3xxs(tab, qs[2 * l + 4]), hh ? s1 >> 4 : s1, sc);
    }
};
template <> struct W8<T_IQ2_XXS> {
    static inline void run(device const uchar* row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        device const uchar* bl = row + (k >> 8) * 66;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint a0 = rd32(bl + 2 + 8 * ib), a1 = rd32(bl + 2 + 8 * ib + 4);
        const float sc = rdh(bl) * (0.5f + float(a1 >> 28)) * 0.25f;
        const uint i0 = (a0 >> (8 * l)) & 255, i1 = (a0 >> (8 * (l + 2))) & 255;
        // through the accessors, so QUANT_CTAB (the constant grids, no threadgroup copy) gives the same values: the
        // direct tab[] reads below skipped the ctab branch and read an unfilled table
        const uint s0 = ksg(tab, 512, (a1 >> (7 * l)) & 127), s1 = ksg(tab, 512, (a1 >> (7 * (l + 2))) & 127);
        a = w4_signed(g_iq2xxs(tab, i0, hh), hh ? s0 >> 4 : s0, sc);
        b = w4_signed(g_iq2xxs(tab, i1, hh), hh ? s1 >> 4 : s1, sc);
    }
};
template <> struct W8<T_IQ2_XS> {
    static inline void run(device const uchar* row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        device const uchar* bl = row + (k >> 8) * 74;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        device const ushort* q = (device const ushort*)(bl + 2) + 4 * ib;
        const uint v0 = q[l], v1 = q[l + 2];
        const uint scb = bl[66 + ib];
        const float d = rdh(bl);
        const float sc0 = d * (0.5f + float((scb >> (4 * (l >> 1))) & 15)) * 0.25f;
        const float sc1 = d * (0.5f + float((scb >> (4 * ((l + 2) >> 1))) & 15)) * 0.25f;
        // through the accessors: the direct tab[]/ks[] reads ignored QUANT_CTAB, whose tables are then never filled
        const uint s0 = ksg(tab, 1024, v0 >> 9), s1 = ksg(tab, 1024, v1 >> 9);
        a = w4_signed(g_iq2xs(tab, v0 & 511, hh), hh ? s0 >> 4 : s0, sc0);
        b = w4_signed(g_iq2xs(tab, v1 & 511, hh), hh ? s1 >> 4 : s1, sc1);
    }
};
template <> struct W8<T_IQ2_S> {
    template <typename P>
    static inline void run(P row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        const P bl = row + (k >> 8) * 82;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint qh = bl[66 + ib], scb = bl[74 + ib];
        const float d = rdh(bl);
        const uint i0 = uint(bl[2 + 4 * ib + l]) | ((qh << (8 - 2 * l)) & 0x300);
        const uint i1 = uint(bl[2 + 4 * ib + l + 2]) | ((qh << (8 - 2 * (l + 2))) & 0x300);
        const uint s0 = bl[2 + 32 + 4 * ib + l], s1 = bl[2 + 32 + 4 * ib + l + 2];
        const float sc0 = d * (0.5f + float((scb >> (4 * (l >> 1))) & 15)) * 0.25f;
        const float sc1 = d * (0.5f + float((scb >> (4 * ((l + 2) >> 1))) & 15)) * 0.25f;
        a = w4_signed(g_iq2s(tab, i0, hh), hh ? s0 >> 4 : s0, sc0);
        b = w4_signed(g_iq2s(tab, i1, hh), hh ? s1 >> 4 : s1, sc1);
    }
};

// IQ1_M (the 2.00 bpw pack's 1.75-bit type), ggml's dequantize_row_iq1_m: the block's F16 scale sits in the top
// nibbles of its four u16 scales; a 32-value sub-block has two 3-bit scales (groups 0-1 and 2-3, eight values each),
// and each eight-value group carries a +-0.125 delta picked by a bit of its qh byte.  The tile's two halves are the
// four values at k and the four at k + 16, each decoded on its own (k + 16 crosses into the next sub-block at l >= 2).
template <typename P>
static inline half4 iq1m_w4(P row, int k) {
    const P bl = row + (k >> 8) * 56;
    const int ib = (k & 255) >> 5, g = (k & 31) >> 3, hh = (k & 7) >> 2;
    const uint s0 = uint(bl[48]) | (uint(bl[49]) << 8), s1 = uint(bl[50]) | (uint(bl[51]) << 8);
    const uint s2 = uint(bl[52]) | (uint(bl[53]) << 8), s3 = uint(bl[54]) | (uint(bl[55]) << 8);
    const float d = float(as_type<half>(ushort((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u))));
    const uint sc = uint(((const device ushort*) (bl + 48))[ib >> 1]);
    const float dl = d * float(2u * ((sc >> (6u * uint(ib & 1) + (g < 2 ? 0u : 3u))) & 7u) + 1u);
    const uint qh = (g < 2) ? uint(bl[32 + 2 * ib]) : uint(bl[33 + 2 * ib]);
    const uint idx = uint(bl[4 * ib + g]) | (((g & 1) != 0) ? ((qh << 4u) & 0x700u) : ((qh << 8u) & 0x700u));
    const bool neg = ((g & 1) != 0) ? ((qh & 0x80u) != 0u) : ((qh & 0x08u) != 0u);
    const ulong gr = iq1s_grid[idx];
    const float delta = neg ? -0.125f : 0.125f;
    float v[4];
    for (int j = 0; j < 4; ++j) {
        const uint by = uint((gr >> (8u * uint(4 * hh + j))) & 0xffu);
        const int c = (by & 0x80u) != 0u ? int(by) - 256 : int(by);
        v[j] = dl * (float(c) + delta);
    }
    return half4(v[0], v[1], v[2], v[3]);
}
template <> struct W8<T_IQ1_M> {
    template <typename P>
    static inline void run(P row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        (void) tab;
        a = iq1m_w4(row, k);
        b = iq1m_w4(row, k + 16);
    }
};

// IQ1_S (Unsloth's UD-IQ1_S / UD-IQ1_M experts), ggml's dequantize_row_iq1_s: f16 d, qs[32], qh[8] u16; a 32-value sub-block has
// one scale dl = d (2 s + 1) (qh bits 12-14), a +-0.125 delta (qh bit 15) and four 8-value groups whose grid index is
// qs[4 ib + g] | 3 bits of qh << 8.  Same float arithmetic as ggml (dl (grid + delta)), so the half4 is what the CPU reference rounds to.
template <typename P>
static inline half4 iq1s_w4(P row, int k) {
    const P bl = row + (k >> 8) * 50;
    const int ib = (k & 255) >> 5, g = (k & 31) >> 3, hh = (k & 7) >> 2;
    const uint qh = uint(bl[34 + 2 * ib]) | (uint(bl[35 + 2 * ib]) << 8);
    const float dl = float(as_type<half>(ushort(uint(bl[0]) | (uint(bl[1]) << 8)))) * float(2u * ((qh >> 12) & 7u) + 1u);
    const float delta = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
    const uint idx = uint(bl[2 + 4 * ib + g]) | (((qh >> (3u * uint(g))) & 7u) << 8);
    const ulong gr = iq1s_grid[idx];
    float v[4];
    for (int j = 0; j < 4; ++j) {
        const uint by = uint((gr >> (8u * uint(4 * hh + j))) & 0xffu);
        const int c = (by & 0x80u) != 0u ? int(by) - 256 : int(by);
        v[j] = dl * (float(c) + delta);
    }
    return half4(v[0], v[1], v[2], v[3]);
}
template <> struct W8<T_IQ1_S> {
    template <typename P>
    static inline void run(P row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        (void) tab;
        a = iq1s_w4(row, k);
        b = iq1s_w4(row, k + 16);
    }
};

template <> struct W8<T_IQ3_S> {
    template <typename P>
    static inline void run(P row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        const P bl = row + (k >> 8) * 110;
        const int ib = (k & 255) >> 5, l = (k & 31) >> 3, hh = (k & 7) >> 2;
        const uint qh = bl[66 + ib], scb = bl[106 + (ib >> 1)];
        const float sc = rdh(bl) * float(1 + 2 * ((scb >> (4 * (ib & 1))) & 15));
        const int l1 = l + 2;
        const uint i0 = hh ? (uint(bl[2 + 8 * ib + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256))
                           : (uint(bl[2 + 8 * ib + 2 * l]) | ((qh << (8 - 2 * l)) & 256));
        const uint i1 = hh ? (uint(bl[2 + 8 * ib + 2 * l1 + 1]) | ((qh << (7 - 2 * l1)) & 256))
                           : (uint(bl[2 + 8 * ib + 2 * l1]) | ((qh << (8 - 2 * l1)) & 256));
        const uint s0 = bl[74 + 4 * ib + l], s1 = bl[74 + 4 * ib + l1];
        if (mm_probe == 7) {
            // bench probe: a repacked layout's cost - one aligned 16-byte load a lane a row a 32-group holding every
            // field the lane needs (wrong values: the bytes come from an arbitrary aligned spot of the row)
            const uint4 r = *(device const uint4*)((ulong(row) + ulong(k >> 5) * 16) & ~ulong(15));
            const uint qhv = r.z & 0xFF, scv = (r.z >> 8) & 0xFF;
            const float scr = float(as_type<half>(ushort(r.w & 0xFFFF))) * float(1 + 2 * ((scv >> (4 * (ib & 1))) & 15));
            const uint j0 = ((r.x >> (8 * (l & 1))) & 0xFF) | ((qhv << (8 - 2 * l)) & 256);
            const uint j1 = ((r.x >> (16 + 8 * (l & 1))) & 0xFF) | ((qhv << (7 - 2 * l)) & 256);
            a = w4_signed(tab[j0], (r.y >> (8 * hh)) & 0xFF, scr);
            b = w4_signed(tab[j1], (r.y >> (16 + 8 * hh)) & 0xFF, scr);
            return;
        }
        if (mm_probe == 5) {   // bench probe: the loads and the scale, no grid lookup / sign math (wrong values)
            a = half4(half(float(i0 ^ s0) * sc));
            b = half4(half(float(i1 ^ s1) * sc));
            return;
        }
        a = w4_signed(g_iq3s(tab, i0), hh ? s0 >> 4 : s0, sc);
        b = w4_signed(g_iq3s(tab, i1), hh ? s1 >> 4 : s1, sc);
    }
};

// W8 for the K-quants and Q5_1: the lane's runs k and k + 16 (k % 32 < 16) share the block's scales / mins / high
// bits; the 4-byte code runs read as one uint where the block layout keeps them 4-aligned (Q4_K 144-, Q5_K 176-,
// Q5_1 24-byte blocks: rows of whole blocks start 4-aligned).  The same per-value arithmetic as W4.
static inline float4 nib4(uint v, int hi) { const uint s = hi ? (v >> 4) : v; return float4(as_type<uchar4>(s & 0x0F0F0F0Fu)); }
template <> struct W8<T_Q4_K> {
    static inline void run(device const uchar* row, int k, threadgroup const uint*, thread half4& a, thread half4& b) {
        device const uchar* bl = row + (k >> 8) * 144;
        const int j = k & 255, c = j >> 6, hi = (j >> 5) & 1, l = j & 31;
        float s, mm;
        scale_min_k4(2 * c + hi, bl + 4, s, mm);
        const float d1 = rdh(bl) * s, m1 = rdh(bl + 2) * mm;
        device const uint* q = (device const uint*)(bl + 16 + 32 * c + l);
        const float4 qa = nib4(q[0], hi), qb = nib4(q[4], hi);
        a = half4(d1 * qa - m1);
        b = half4(d1 * qb - m1);
    }
};
template <> struct W8<T_Q5_K> {
    static inline void run(device const uchar* row, int k, threadgroup const uint*, thread half4& a, thread half4& b) {
        device const uchar* bl = row + (k >> 8) * 176;
        const int j = k & 255, c = j >> 6, hi = (j >> 5) & 1, l = j & 31;
        float s, mm;
        scale_min_k4(2 * c + hi, bl + 4, s, mm);
        const float d1 = rdh(bl) * s, m1 = rdh(bl + 2) * mm;
        device const uint* qh = (device const uint*)(bl + 16 + l);
        device const uint* ql = (device const uint*)(bl + 48 + 32 * c + l);
        const uint sh = uint(2 * c + hi);
        const float4 ha = float4(as_type<uchar4>((qh[0] >> sh) & 0x01010101u)) * 16.0f;
        const float4 hb = float4(as_type<uchar4>((qh[4] >> sh) & 0x01010101u)) * 16.0f;
        a = half4(d1 * (nib4(ql[0], hi) + ha) - m1);
        b = half4(d1 * (nib4(ql[4], hi) + hb) - m1);
    }
};
template <> struct W8<T_Q5_1> {
    static inline void run(device const uchar* row, int k, threadgroup const uint*, thread half4& a, thread half4& b) {
        device const uchar* bl = row + (k >> 5) * 24;
        const float d = rdh(bl), m = rdh(bl + 2);
        const uint hb = rd32(bl + 4);
        const int j = k & 15;   // runs at j and j + 16: the low nibbles of qs[j..j+3], then the high ones
        const uint v = *(device const uint*)(bl + 8 + j);
        float4 wa, wb;
        for (int c = 0; c < 4; ++c) {
            const uint qv = (v >> (8 * c)) & 0xFFu;
            wa[c] = float(int(qv & 15) | int(((hb >> (j + c)) & 1) << 4)) * d + m;
            wb[c] = float(int(qv >> 4) | int(((hb >> (j + 16 + c)) & 1) << 4)) * d + m;
        }
        a = half4(wa);
        b = half4(wb);
    }
};
template <> struct W8<T_Q6_K> {
    static inline void run(device const uchar* row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        device const uchar* bl = row + (k >> 8) * 210;
        const int j = k & 255, n = j >> 7, jj = j & 127, qd = jj >> 5, l = jj & 31;   // l < 16; the pair at l + 16
        device const uchar* ql = bl + 64 * n + l + ((qd & 1) ? 32 : 0);
        device const uchar* qh = bl + 128 + 32 * n + l;
        const float d = rdh(bl + 208);
        device const char* sc = (device const char*)(bl + 192 + 8 * n);
        const float dsa = d * float(sc[2 * qd]), dsb = d * float(sc[1 + 2 * qd]);
        // the 4-byte runs as two ushorts each (210-byte blocks: 2-aligned)
        device const ushort* l2 = (device const ushort*)ql;
        device const ushort* h2 = (device const ushort*)qh;
        const uint la = uint(l2[0]) | (uint(l2[1]) << 16), lb = uint(l2[8]) | (uint(l2[9]) << 16);
        const uint ha = uint(h2[0]) | (uint(h2[1]) << 16), hb = uint(h2[8]) | (uint(h2[9]) << 16);
        const uint sl = (qd >> 1) ? 4u : 0u, shh = uint(2 * qd);
        const uint4 qa = (uint4(as_type<uchar4>((la >> sl) & 0x0F0F0F0Fu)) | (uint4(as_type<uchar4>((ha >> shh) & 0x03030303u)) << 4));
        const uint4 qb = (uint4(as_type<uchar4>((lb >> sl) & 0x0F0F0F0Fu)) | (uint4(as_type<uchar4>((hb >> shh) & 0x03030303u)) << 4));
        a = half4(dsa * (float4(qa) - 32.0f));
        b = half4(dsb * (float4(qb) - 32.0f));
    }
};

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_gu3(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]],
                   device half* HS [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]], device const uint* smap [[buffer(5)]],
                     device const uchar* W1 [[buffer(6)]], device const uchar* W2 [[buffer(7)]], device const uchar* W3 [[buffer(8)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* E = kArena ? arena_slot(smap[e], W, W1, W2, W3) : W + ulong(e) * a.gu_stride;
    device const uchar* g0 = E + a.gate_off + ulong(n0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(n0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    auto rg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ru = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    auto accu = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        const uint16_t cap = rg.get_capacity(), half_cap = cap / 2;
        if (!coop_k4) {
            for (uint16_t i = 0; i < cap; ++i) {
                const auto ix = rg.get_multidimensional_index(i);
                const int k = k0 + ix[0], n = ix[1];
                rg[i] = W4<F>::run(g0 + ulong(n) * a.gu_row, k & ~3, tab)[k & 3];
                ru[i] = W4<F>::run(u0 + ulong(n) * a.gu_row, k & ~3, tab)[k & 3];
            }
        } else
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rg.get_multidimensional_index(i);
            const auto iy = rg.get_multidimensional_index(i + half_cap);
            const int k = k0 + ix[0], n = ix[1];
            device const uchar* rgw = g0 + ulong(n) * a.gu_row;
            device const uchar* ruw = u0 + ulong(n) * a.gu_row;
            half4 wg, wu, wg2, wu2;
            if (mm_probe == 1) {   // the bench probe: no decode (constant weights)
                wg = half4(half(n & 7)); wu = wg; wg2 = half4(half(k & 7)); wu2 = wg2;
            } else if (TK == 32) {
                // the lane's two runs of this row sit in one 32-group: element i + cap / 2 is (k + 16, n) - part of
                // the COOP_K4 layout (coop_probe checks it); a runtime check of it here cost 30%
                W8<F>::run(rgw, k, tab, wg, wg2);
                W8<F>::run(ruw, k, tab, wu, wu2);
            } else {
                const int k2 = k0 + iy[0];
                device const uchar* rgw2 = g0 + ulong(iy[1]) * a.gu_row;
                device const uchar* ruw2 = u0 + ulong(iy[1]) * a.gu_row;
                wg = W4<F>::run(rgw, k, tab); wu = W4<F>::run(ruw, k, tab);
                wg2 = W4<F>::run(rgw2, k2, tab); wu2 = W4<F>::run(ruw2, k2, tab);
            }
            rg[i] = wg.x; rg[i + 1] = wg.y; rg[i + 2] = wg.z; rg[i + 3] = wg.w;
            ru[i] = wu.x; ru[i + 1] = wu.y; ru[i + 2] = wu.z; ru[i + 3] = wu.w;
            const uint16_t j = i + half_cap;
            rg[j] = wg2.x; rg[j + 1] = wg2.y; rg[j + 2] = wg2.z; rg[j + 3] = wg2.w;
            ru[j] = wu2.x; ru[j + 1] = wu2.y; ru[j + 2] = wu2.z; ru[j + 3] = wu2.w;
        }
        tA = tX.template slice<TK, TM>(k0, row0);
        op.run(tA, rg, accg);
        op.run(tA, ru, accu);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = accg[i];
            accg[i] = g / (1.0f + exp(-g)) * accu[i];
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        HS[ulong(row0 + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_down3(device const uchar* W [[buffer(0)]], device const half* HS [[buffer(1)]],
                     device float* YE [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]], device const uint* smap [[buffer(5)]],
                     device const uchar* W1 [[buffer(6)]], device const uchar* W2 [[buffer(7)]], device const uchar* W3 [[buffer(8)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* w0 = (kArena ? arena_slot(smap[e], W, W1, W2, W3) : W + ulong(e) * a.d_stride) + a.down_off + ulong(n0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    auto rw = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rw)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        if (!coop_k4) {
            for (uint16_t i = 0; i < rw.get_capacity(); ++i) {
                const auto ix = rw.get_multidimensional_index(i);
                const int k = k0 + ix[0];
                rw[i] = W4<F>::run(w0 + ulong(ix[1]) * a.d_row, k & ~3, tab)[k & 3];
            }
        } else if (TK == 32) {
            // the paired runs (element i + cap / 2 is (k + 16, n), the COOP_K4 layout): one W8 a pair
            const uint16_t half_cap = rw.get_capacity() / 2;
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < half_cap; i += 4) {
                const auto ix = rw.get_multidimensional_index(i);
                half4 w, w2;
                W8<F>::run(w0 + ulong(ix[1]) * a.d_row, k0 + ix[0], tab, w, w2);
                rw[i] = w.x; rw[i + 1] = w.y; rw[i + 2] = w.z; rw[i + 3] = w.w;
                const uint16_t j = i + half_cap;
                rw[j] = w2.x; rw[j + 1] = w2.y; rw[j + 2] = w2.z; rw[j + 3] = w2.w;
            }
        } else
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rw.get_capacity(); i += 4) {
            const auto ix = rw.get_multidimensional_index(i);
            const half4 w = W4<F>::run(w0 + ulong(ix[1]) * a.d_row, k0 + ix[0], tab);
            rw[i] = w.x; rw[i + 1] = w.y; rw[i + 2] = w.z; rw[i + 3] = w.w;
        }
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, rw, acc);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    acc.store(tY.slice(n0, row0));
}

#define MMGU3_(F, M, N, K, S) template [[host_name("mm_gu3_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_gu3<F, M, N, K, S>) mm_gu3<F, M, N, K, S>;
#define MMDN3_(F, M, N, K, S) template [[host_name("mm_down3_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_down3<F, M, N, K, S>) mm_down3<F, M, N, K, S>;
#define MMGU3(F, M, N, K, S) MMGU3_(F, M, N, K, S)
#define MMDN3(F, M, N, K, S) MMDN3_(F, M, N, K, S)
#define MM3_ONE(F, M, N, K, S) MMGU3(F, M, N, K, S) MMDN3(F, M, N, K, S)
#define MM3_GU(F) MMGU3(F, 16, 64, 32, 2) MMGU3(F, 16, 32, 32, 4) MMGU3(F, 32, 64, 32, 2) MMGU3(F, 16, 64, 64, 2) \
    MMGU3(F, 16, 32, 16, 2) MMGU3(F, 32, 32, 32, 2) MMGU3(F, 16, 64, 32, 1) MMGU3(F, 16, 64, 32, 4)     MMGU3(F, 16, 32, 16, 4) MMGU3(F, 16, 32, 32, 5) MMGU3(F, 16, 32, 16, 5) MMGU3(F, 16, 16, 32, 4) MMGU3(F, 16, 16, 32, 8)     MMGU3(F, 16, 16, 16, 8)
#define MM3_DN(F) MMDN3(F, 16, 64, 32, 2) MMDN3(F, 16, 32, 32, 4) MMDN3(F, 32, 64, 32, 2) MMDN3(F, 16, 64, 64, 2) \
    MMDN3(F, 16, 32, 16, 2) MMDN3(F, 32, 32, 32, 2) MMDN3(F, 16, 64, 32, 1) MMDN3(F, 16, 64, 32, 4)     MMDN3(F, 16, 64, 16, 2) MMDN3(F, 16, 32, 16, 4) MMDN3(F, 16, 64, 32, 5) MMDN3(F, 16, 64, 16, 4)
MM3_GU(T_Q4_K) MM3_GU(T_Q5_K) MM3_GU(T_Q6_K) MM3_GU(T_Q8_0) MM3_GU(T_IQ4_XS) MM3_GU(T_Q2_0) MM3_GU(T_IQ3_XXS) MM3_GU(T_IQ2_XXS) MM3_GU(T_IQ2_XS) MM3_GU(T_IQ2_S) MM3_GU(T_IQ3_S)
MM3_DN(T_Q2_0) MM3_DN(T_IQ4_NL) MM3_DN(T_Q5_1) MM3_DN(T_Q8_0)

// mm_gu4: mm_gu3 with each decoded right operand (TN x TK weights of gate and of up) applied to MS 16-row token
// tiles (a tile of 16 * MS rows from the plan, tm = 16 * MS): the decode, which bounds mm_gu3, is shared by MS times
// as many token rows (more padding rows per expert, but the matrix ops are cheap next to the decode).  The same
// per-row arithmetic as mm_gu3 (each 16-row op is mm_gu3's): bitwise for the rows both compute.  COOP_K4 layout only.
template <int F, int MS, int TN, int TK, int SG>
kernel void mm_gu4(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]],
                   device half* HS [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* g0 = W + a.gate_off + ulong(e) * a.gu_stride + ulong(n0) * a.gu_row;
    device const uchar* u0 = W + a.up_off + ulong(e) * a.gu_stride + ulong(n0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    auto rg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ru = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA0 = tX.template slice<TK, TM>(0, row0);
    auto tA1 = tX.template slice<TK, TM>(0, row0 + TM);
#define MM4_DST op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>,                                                                __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>()
    auto accg0 = MM4_DST;
    auto accu0 = MM4_DST;
    auto accg1 = MM4_DST;
    auto accu1 = MM4_DST;
#undef MM4_DST
    using DT = decltype(accg0);
    for (uint16_t i = 0; i < accg0.get_capacity(); ++i)
        if (accg0.is_valid_element(i)) { accg0[i] = 0.0f; accu0[i] = 0.0f; accg1[i] = 0.0f; accu1[i] = 0.0f; }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        const uint16_t cap = rg.get_capacity(), half_cap = cap / 2;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rg.get_multidimensional_index(i);
            const int k = k0 + ix[0], n = ix[1];
            device const uchar* rgw = g0 + ulong(n) * a.gu_row;
            device const uchar* ruw = u0 + ulong(n) * a.gu_row;
            half4 wg, wu, wg2, wu2;
            W8<F>::run(rgw, k, tab, wg, wg2);
            W8<F>::run(ruw, k, tab, wu, wu2);
            rg[i] = wg.x; rg[i + 1] = wg.y; rg[i + 2] = wg.z; rg[i + 3] = wg.w;
            ru[i] = wu.x; ru[i + 1] = wu.y; ru[i + 2] = wu.z; ru[i + 3] = wu.w;
            const uint16_t j = i + half_cap;
            rg[j] = wg2.x; rg[j + 1] = wg2.y; rg[j + 2] = wg2.z; rg[j + 3] = wg2.w;
            ru[j] = wu2.x; ru[j + 1] = wu2.y; ru[j + 2] = wu2.z; ru[j + 3] = wu2.w;
        }
        tA0 = tX.template slice<TK, TM>(k0, row0);
        op.run(tA0, rg, accg0);
        op.run(tA0, ru, accu0);
        if (MS > 1) {
            tA1 = tX.template slice<TK, TM>(k0, row0 + TM);
            op.run(tA1, rg, accg1);
            op.run(tA1, ru, accu1);
        }
    }
    for (int s = 0; s < MS; ++s) {
        thread DT& ag = s == 0 ? accg0 : accg1;
        thread DT& au = s == 0 ? accu0 : accu1;
        for (uint16_t i = 0; i < ag.get_capacity(); ++i)
            if (ag.is_valid_element(i)) {
                const float g = ag[i];
                ag[i] = g / (1.0f + exp(-g)) * au[i];
            }
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
        simdgroup_barrier(mem_flags::mem_threadgroup);
        ag.store(tO);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int it = lane; it < TM * TN; it += 32) {
            const int m = it / TN, f = it - m * TN;
            HS[ulong(row0 + s * TM + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
        }
    }
}
#define MMGU4_(F, M, N, K, S) template [[host_name("mm_gu4_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_gu4<F, M, N, K, S>) mm_gu4<F, M, N, K, S>;
#define MMGU4(F, M, N, K, S) MMGU4_(F, M, N, K, S)
#define MM4_GU(F) MMGU4(F, 2, 32, 32, 4) MMGU4(F, 2, 32, 32, 2) MMGU4(F, 2, 16, 32, 4) MMGU4(F, 1, 32, 32, 4)
MM4_GU(T_Q2_0) MM4_GU(T_IQ3_XXS) MM4_GU(T_IQ2_XXS) MM4_GU(T_IQ2_XS) MM4_GU(T_IQ2_S) MM4_GU(T_IQ3_S) MM4_GU(T_IQ4_XS)

// mm_gu5: mm_gu3 (COOP_K4 layout, TK 32) with the threadgroup's weight rows staged block by block (QK = 256 values a
// row, gate and up) into threadgroup memory by coalesced ushort loads, the lanes decoding from there: the decode's
// many small scattered device loads (2-6 a run) become threadgroup reads.  Same decoded values and the same matrix
// ops as mm_gu3: bitwise.  Formats with 256-value blocks and a templated W8 (IQ3_S, IQ3_XXS, IQ2_S).
template <int F, int TN, int SG>
kernel void mm_gu5(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]],
                   device half* HS [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16, TK = 32, BB = fmt<F>::BB, ROWS = SG * TN, RU = BB / 2;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup ushort stg[2][ROWS * RU];
    load_tab<F>(tab, tid, SG * 32);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, nt0 = int(tg.x) * ROWS, n0 = nt0 + sg * TN;
    device const uchar* gt = W + a.gate_off + ulong(e) * a.gu_stride + ulong(nt0) * a.gu_row;
    device const uchar* ut = W + a.up_off + ulong(e) * a.gu_stride + ulong(nt0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    auto rg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ru = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    auto accu = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    const int NB = int(a.n_embd) / 256;
    for (int kb = 0; kb < NB; ++kb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < ROWS * RU; idx += SG * 32) {
            const int r = idx / RU, w = idx - r * RU;
            stg[0][idx] = ((device const ushort*)(gt + ulong(r) * a.gu_row + kb * BB))[w];
            stg[1][idx] = ((device const ushort*)(ut + ulong(r) * a.gu_row + kb * BB))[w];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int ks = 0; ks < 8; ++ks) {
            const int k0 = kb * 256 + ks * TK;
            const uint16_t half_cap = rg.get_capacity() / 2;
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < half_cap; i += 4) {
                const auto ix = rg.get_multidimensional_index(i);
                const int kl = ks * TK + ix[0], n = sg * TN + ix[1];
                // the block's row in the stage; W8 indexes block (k >> 8) = 0 of it with k = kl
                threadgroup const uchar* rgw = (threadgroup const uchar*)(stg[0] + n * RU);
                threadgroup const uchar* ruw = (threadgroup const uchar*)(stg[1] + n * RU);
                half4 wg, wu, wg2, wu2;
                W8<F>::run(rgw, kl, tab, wg, wg2);
                W8<F>::run(ruw, kl, tab, wu, wu2);
                rg[i] = wg.x; rg[i + 1] = wg.y; rg[i + 2] = wg.z; rg[i + 3] = wg.w;
                ru[i] = wu.x; ru[i + 1] = wu.y; ru[i + 2] = wu.z; ru[i + 3] = wu.w;
                const uint16_t j = i + half_cap;
                rg[j] = wg2.x; rg[j + 1] = wg2.y; rg[j + 2] = wg2.z; rg[j + 3] = wg2.w;
                ru[j] = wu2.x; ru[j + 1] = wu2.y; ru[j + 2] = wu2.z; ru[j + 3] = wu2.w;
            }
            tA = tX.template slice<TK, TM>(k0, row0);
            op.run(tA, rg, accg);
            op.run(tA, ru, accu);
        }
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = accg[i];
            accg[i] = g / (1.0f + exp(-g)) * accu[i];
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        HS[ulong(row0 + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
    }
}
#define MMGU5_(F, N, S) template [[host_name("mm_gu5_" #F "_" #N "_" #S)]] kernel decltype(mm_gu5<F, N, S>) mm_gu5<F, N, S>;
#define MMGU5(F, N, S) MMGU5_(F, N, S)
MMGU5(T_IQ3_S, 32, 2) MMGU5(T_IQ3_S, 32, 1) MMGU5(T_IQ3_S, 16, 4) MMGU5(T_IQ3_XXS, 32, 2) MMGU5(T_IQ3_XXS, 32, 1)
MMGU5(T_IQ3_XXS, 16, 4) MMGU5(T_IQ2_S, 32, 2) MMGU5(T_IQ2_S, 32, 1) MMGU5(T_IQ2_S, 16, 4)
