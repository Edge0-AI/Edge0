// src/prefill/gdn_tc.cuh - the chunked DeltaNet recurrence's scan on the tensor cores (STRATA_GDN_TC=1), sm_80+.
//
// Included inside the anonymous namespace of src/prefill/kernels.cu and of src/prefill/gdn_rec_parity.cu, after
// gdn_chunk_scan_kernel (the harness runs the engine's kernels verbatim: one text, included twice).  It uses S, HK, HV,
// C, VPK, GCH, GSB, GTP, gdn_cp16 / gdn_cp_commit / gdn_cp_wait_all from there and reads no environment.
//
// gdn_chunk_scan_kernel's chunk algebra (kernels.cu, "STRATA_GDN_CHUNKED=1"), every product transposed so that a
// warp's slice of the state is the mma A operand and never leaves its registers: a block per value head, warp w owns
// value columns 16w..16w+15 of S^T (16 columns x 128 rows = 16 m16n8 f32 accumulator tiles, 64 registers a lane) and
// all 32 tokens of a chunk, so not one product couples two warps:
//   X^T  (16c x 32i) = S0^T . K^T        f16x3, k = state row (8 k16-steps), n = token (4 tiles)
//   QS^T (16c x 32t) = S0^T . Q^T        the same A fragments
//   Y^T  (16c x 32i) = beta_i (V^T - e_i X^T)                                     elementwise on the accumulators
//   D^T  (16c x 32i) = Y^T . T^T         f16x3, k = token i' (2 steps), n = token i: T is zero above its diagonal,
//                                        so 6 of 8 tiles (as P below)
//   o^T  (16c x 32t) = e_t QS^T + D^T . P^T   f16x3, k = i (2 steps), n = t: P zero above its diagonal (6 of 8 tiles)
//   Dd^T (16c x 32i) = f_i D^T,  f_i = exp(gL - gamma_i)
//   S^T  (16c x 128r) = eL S0^T + Dd^T . K    f16x3, k = token (2 steps), n = state row (16 tiles)
// 324 mma a warp a chunk; the only shared data are the staged inputs (k, q, T and P as f16 hi / lo images, v, the
// per-token scalars), one barrier a chunk.
//
// Numerics (the harness holds the result within 1e-4 of gdn_rec_kh_kernel, as the FP32 chunked pair): an f16 operand
// is split x = hi + lo, hi = f16(x), lo = f16(x - hi), and a product takes hi.hi + hi.lo + lo.hi, dropping lo.lo: with
// both parts normal that is within 3 x 2^-22 |a||b| of the exact product, the 3xTF32 bound, at half the tensor time.
// For the low part to be normal the operand's largest entry must sit in [2^14, 2^15), which a power of two does
// exactly: k and q (unit vectors after gdn_l2_kernel) carry 2^8, P (|P| <= 1) 2^8, the state 2^es from a warp max
// per chunk, Y^T 2^ey and D^T 2^ed likewise, T = (I + A)^-1 (no a-priori bound) 2^et from the prep's warp max; the
// powers are taken out of the per-token scalars and of the accumulator before and after the mma (one FMUL an
// element, exact).  (D^T = Y^T T^T on tf32x3 with T split at use, the form before this one, cost 30 mma, 32 shuffles
// and 48 tf32 conversions a warp a chunk against 18 mma now; measured in the commit that changed it.)
#if !defined(__HIPCC__)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_GDN_TC_SM80 1
#else
#define STRATA_GDN_TC_SM80 0   // device code for older cards: the wrappers trap (gdn_rec_tc never launches there)
#endif

constexpr int GTC_KP = S + 8;     // f16 k / q rows: a 272-B stride puts the 8 rows of a ldmatrix 8x8 on 8 distinct 16-B lanes
constexpr int GTC_VP = S + 4;     // f32 v rows: lane (gid, tig) reads word 264 tig + gid, 32 distinct banks
constexpr int GTC_PP = GCH + 8;   // f16 T / P rows: an 80-B stride, 8 rows on distinct 16-B lanes
constexpr int GTC_CW = S / 16;    // consumer warps a block: 16 value columns each
// dynamic shared memory a block, two buffers each: k / q hi / lo, v, (z), T hi / lo, P hi / lo, the token scalars
// (beta, exp(gamma), exp(gL - gamma), gamma) as float4, (the norm's o^2 partials), T's scale 2^-et: 124,944 B
// without the fused norm, 160,784 with it (the 166,912 opt-in of GA100)
template <bool NORM>
constexpr size_t kTcScanSmem = 2 * (4 * GCH * GTC_KP * 2 + GCH * GTC_VP * 4 + (NORM ? GCH * GTC_VP * 4 : 0) +
                                    4 * GCH * GTC_PP * 2 + GCH * 16 + (NORM ? GCH * GTC_CW * 4 : 0)) + 16;
static_assert(kTcScanSmem<false> == 124944 && kTcScanSmem<true> == 160784, "gdn_tc.cuh: the shared-memory plan");
// SPLIT (the 99-KB cards: sm_86 / sm_89 / sm_120): only k / q hi / lo and the token scalars stay double-buffered; v, T and P
// (one buffer) are staged for the next chunk after this chunk's last read of them, so that load overlaps the state update:
// 97,808 B against the 101,376 B opt-in of those cards (the full plan needs 124,944)
constexpr size_t kTcScanSmemSplit = 2 * (4 * GCH * GTC_KP * 2 + GCH * 16) + GCH * GTC_VP * 4 + 4 * GCH * GTC_PP * 2 + 16;
static_assert(kTcScanSmemSplit == 97808, "gdn_tc.cuh: the split shared-memory plan");
// + the fused output norm's partial sums of squares [GTC_CW warps][GCH tokens] (SPLIT only): 98,832 B
constexpr size_t kTcScanSmemSplitNorm = kTcScanSmemSplit + GTC_CW * GCH * 4;
static_assert(kTcScanSmemSplitNorm == 98832, "gdn_tc.cuh: the split + norm shared-memory plan");
// the prep's scratch for a super-block (gdn_tc_scratch, one a device, 60.3 MB): the T images x 2^et as
// [chunk][vh][2: hi, lo][GCH][GCH] f16 and their scales 2^-et as [chunk][vh], the token scalars as float4
// [chunk][vh][GCH], the k and q images x 2^8 as [chunk][qh][4: k hi, k lo, q hi, q lo][GCH][S] and the P images x 2^8
// as [chunk][vh][2: hi, lo][GCH][GCH] - every row a multiple of 16 B, so the scan stages all of it by cp.async
constexpr int GTC_NB = GSB / GCH;
struct GtcScratch {
    uint16_t* ti;
    float* te;
    float4* sc;
    uint16_t* kq;
    uint16_t* pi;
};
constexpr size_t kTcTiBytes = (size_t) GTC_NB * HV * 2 * GCH * GCH * 2, kTcTeBytes = (size_t) GTC_NB * HV * 4,
                 kTcScBytes = (size_t) GTC_NB * HV * GCH * 16, kTcKqBytes = (size_t) GTC_NB * HK * 4 * GCH * S * 2,
                 kTcPiBytes = (size_t) GTC_NB * HV * 2 * GCH * GCH * 2;
static_assert(kTcTiBytes + kTcTeBytes + kTcScBytes + kTcKqBytes + kTcPiBytes == 60305408, "gdn_tc.cuh: the scratch plan");
// allocated on first use and kept, as gdn_chunk_scratch (kernels.cu); ti == nullptr when the card had no room
GtcScratch gdn_tc_scratch() {
    static unsigned char* bufs[64] = {};
    GtcScratch r = {nullptr, nullptr, nullptr, nullptr, nullptr};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return r; }
    if (bufs[dev] == nullptr && cudaMalloc((void**) &bufs[dev], kTcTiBytes + kTcTeBytes + kTcScBytes + kTcKqBytes + kTcPiBytes) != cudaSuccess) {
        cudaGetLastError();
        bufs[dev] = nullptr;
        return r;
    }
    unsigned char* p = bufs[dev];
    r.ti = reinterpret_cast<uint16_t*>(p);
    r.te = reinterpret_cast<float*>(p + kTcTiBytes);
    r.sc = reinterpret_cast<float4*>(p + kTcTiBytes + kTcTeBytes);
    r.kq = reinterpret_cast<uint16_t*>(p + kTcTiBytes + kTcTeBytes + kTcScBytes);
    r.pi = reinterpret_cast<uint16_t*>(p + kTcTiBytes + kTcTeBytes + kTcScBytes + kTcKqBytes);
    return r;
}

// the warp's largest of non-negative floats: their bits order as unsigned integers, so one redux.sync (sm_80).  Under
// the full mask: __activemask() does not reconverge the warp (CUDA guide), and the first call of the scan follows a
// lane-divergent branch (the L2 prefetch of tokens past the end, in the chunk two before the end whenever
// 0 < T - f_t0 < 32), so with it the lanes could reduce under a partial mask and disagree on es.  The harness build
// happened to close that branch (BSYNC) before the REDUX.MAX; the engine's build is a separate compile that nothing
// checks.  The kernel has no early return, so the full mask costs no reconvergence code.
__device__ __forceinline__ float gtc_warp_max(float v) {
#if STRATA_GDN_TC_SM80
    return __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(v)));
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
#endif
}
__device__ __forceinline__ float gtc_pow2(int e) { return __int_as_float((e + 127) << 23); }   // 2^e, |e| < 127
// the power of two that puts m = max |x| >= 0 in [2^14, 2^15): f16 hi never overflows (65504) and the low part of
// every entry within 2^-17 of the max stays normal; clamped so that the compensating powers stay far from f32's range
__device__ __forceinline__ int gtc_scale_exp(float m) {
    if (m == 0.0f) return 0;
    const int e = 14 - ((__float_as_int(m) >> 23) - 127);
    return e < -40 ? -40 : (e > 40 ? 40 : e);
}

// ---- the tensor-core wrappers (PTX ISA "Matrix Fragments": lane = 4 gid + tig)
//   m16n8k16 f16: A a0 = (row gid, cols 2tig, 2tig+1) a1 = (gid+8, ..) a2 = (gid, 2tig+8..) a3 = (gid+8, 2tig+8..);
//                 B b0 = (k 2tig, 2tig+1; n gid) b1 = (k 2tig+8..; n gid); C/D c0 = (gid, 2tig) c1 = (gid, 2tig+1)
//                 c2 = (gid+8, 2tig) c3 = (gid+8, 2tig+1) - so two adjacent C tiles ARE the next product's A fragment
//   ldmatrix m8n8.x4.b16: lanes 8m..8m+7 give the row addresses of matrix m; a lane gets (row gid, cols 2tig, 2tig+1)
//                 of each matrix (the lower column in the low half), or with .trans (rows 2tig, 2tig+1; col gid)
__device__ __forceinline__ void gtc_ldsm_x4(uint32_t (&r)[4], const void* smem) {
#if STRATA_GDN_TC_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u; (void) smem; __trap();
#endif
}
__device__ __forceinline__ void gtc_ldsm_x4_trans(uint32_t (&r)[4], const void* smem) {
#if STRATA_GDN_TC_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u; (void) smem; __trap();
#endif
}
__device__ __forceinline__ void gtc_mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if STRATA_GDN_TC_SM80
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void) c; (void) a; (void) b0; (void) b1; __trap();
#endif
}
// the three passes of a split product, small terms first (as qsa_select.cu): lo.hi, hi.lo, hi.hi
__device__ __forceinline__ void gtc_mma3_f16(float (&c)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0,
                                             uint32_t bh1, uint32_t bl0, uint32_t bl1) {
    gtc_mma_f16(c, al, bh0, bh1);
    gtc_mma_f16(c, ah, bl0, bl1);
    gtc_mma_f16(c, ah, bh0, bh1);
}
// NP (experiment, STRATA_GDN_TC split kernels only): 3 = every product on the three passes above (the default); 2 = only the products
// that reach the OUTPUT alone (q.S and D.P) on one pass, the state's products (k.S, Y.T, the update) on three; 1 = every product on one pass
// (hi.hi, plain f16 operands as the gfx11 WMMA path).  A single-pass product does not read its B operand's lo image.
template <int NP>
__device__ __forceinline__ void gtc_mmaN(float (&c)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0,
                                         uint32_t bh1, uint32_t bl0, uint32_t bl1) {
    if constexpr (NP == 3) {
        gtc_mma_f16(c, al, bh0, bh1);
        gtc_mma_f16(c, ah, bl0, bl1);
    } else if constexpr (NP == 2) {
        gtc_mma_f16(c, al, bh0, bh1);
    }
    gtc_mma_f16(c, ah, bh0, bh1);
}
// x0, x1 -> packed f16 hi (x0 in the low half) and lo = f16(x - hi): x - hi is exact in f32 (hi carries x's exponent)
__device__ __forceinline__ void gtc_split_f16x2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
    union { __half2 h; uint32_t u; } cv;
    cv.h = __floats2half2_rn(x0, x1);
    hi = cv.u;
    const float2 r = __half22float2(cv.h);
    cv.h = __floats2half2_rn(x0 - r.x, x1 - r.y);
    lo = cv.u;
}
// four consecutive elements of an f16 hi / lo row from a float4 x scale (an 8-byte store each)
__device__ __forceinline__ void gtc_split_store4(uint16_t* hi, uint16_t* lo, float4 v, float scale) {
    uint32_t h0, l0, h1, l1;
    gtc_split_f16x2(v.x * scale, v.y * scale, h0, l0);
    gtc_split_f16x2(v.z * scale, v.w * scale, h1, l1);
    *reinterpret_cast<uint2*>(hi) = make_uint2(h0, h1);
    *reinterpret_cast<uint2*>(lo) = make_uint2(l0, l1);
}
// a float4 from shared memory the compiler does not keep: the per-token scalars are read in three phases of a chunk,
// a component each, and kept across them they cost 24 registers a thread
__device__ __forceinline__ float4 gtc_lds128(const float4* p) {
    float4 v;
#if STRATA_GDN_TC_SM80
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
#else
    v = *p;
#endif
    return v;
}
// ---- the products of a chunk, a warp each (the self-test of gdn_rec_parity --selftest-frag runs them one by one)
// X^T = S^T K^T and QS^T = S^T Q^T: A = the state fragments x sA (= 2^es) split in f16 hi / lo just in time per k16-step
// (tiles 2kk and 2kk + 1 of S^T are that step's A fragment), B = the f16 hi / lo images of k and q ([GCH][GTC_KP] in
// shared memory, rows = tokens) by ldmatrix: matrices (n-tile 2hf + (lane >> 4), k-half (lane >> 3) & 1), a lane's
// row is token 8 n-tile + (lane & 7).  The accumulators come out as 2^(8 + es) times the products.  2 x 96 mma.
template <int NPK = 3, int NPQ = 3>
__device__ __forceinline__ void gtc_prod_state(const float (&s)[16][4], float sA, const uint16_t* kh, const uint16_t* kl,
                                               const uint16_t* qh, const uint16_t* ql, float (&xt)[4][4], float (&qs)[4][4]) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * GTC_KP + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int kk = 0; kk < S / 16; ++kk) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(s[2 * kk][0] * sA, s[2 * kk][1] * sA, ah[0], al[0]);
        gtc_split_f16x2(s[2 * kk][2] * sA, s[2 * kk][3] * sA, ah[1], al[1]);
        gtc_split_f16x2(s[2 * kk + 1][0] * sA, s[2 * kk + 1][1] * sA, ah[2], al[2]);
        gtc_split_f16x2(s[2 * kk + 1][2] * sA, s[2 * kk + 1][3] * sA, ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int o = off + 16 * hf * GTC_KP + 16 * kk;
            uint32_t bh[4], bl[4] = {0u, 0u, 0u, 0u};
            gtc_ldsm_x4(bh, kh + o);
            if constexpr (NPK == 3) gtc_ldsm_x4(bl, kl + o);
            gtc_mmaN<NPK>(xt[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mmaN<NPK>(xt[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            gtc_ldsm_x4(bh, qh + o);
            if constexpr (NPQ == 3) gtc_ldsm_x4(bl, ql + o);
            gtc_mmaN<NPQ>(qs[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mmaN<NPQ>(qs[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}
// o^T += (sD D^T) (2^8 P^T) on f16x3: A = the D^T tiles x sD (= 2^ed) split per k16-step, B = the f16 hi / lo images
// of P x 2^8 ([GCH][GTC_PP], rows = tokens t) by ldmatrix as in gtc_prod_state.  SKIP leaves out the tiles above P's
// diagonal: tokens t < 16 (n-tiles 0, 1) see no i >= 16 (k-step 1).  18 (SKIP) or 24 mma.  D^T += (sY Y^T) (2^et T^T)
// is the same product on the T images (T is zero above its diagonal too), the accumulator x 2^(ey + et).
template <bool SKIP, int NP = 3>
__device__ __forceinline__ void gtc_prod_dt(const float (&dt)[4][4], float sD, const uint16_t* ph, const uint16_t* pl,
                                            float (&ot)[4][4]) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * GTC_PP + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int ks = 0; ks < GCH / 16; ++ks) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(dt[2 * ks][0] * sD, dt[2 * ks][1] * sD, ah[0], al[0]);
        gtc_split_f16x2(dt[2 * ks][2] * sD, dt[2 * ks][3] * sD, ah[1], al[1]);
        gtc_split_f16x2(dt[2 * ks + 1][0] * sD, dt[2 * ks + 1][1] * sD, ah[2], al[2]);
        gtc_split_f16x2(dt[2 * ks + 1][2] * sD, dt[2 * ks + 1][3] * sD, ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            if (SKIP && ks == 1 && hf == 0) continue;
            const int o = off + 16 * hf * GTC_PP + 16 * ks;
            uint32_t bh[4], bl[4] = {0u, 0u, 0u, 0u};
            gtc_ldsm_x4(bh, ph + o);
            if constexpr (NP == 3) gtc_ldsm_x4(bl, pl + o);
            gtc_mmaN<NP>(ot[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mmaN<NP>(ot[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}
// S^T += Dd^T (2^8 K) on f16x3: A = the (already scaled) Dd^T tiles split per k16-step, B = the same k images read the
// other way by ldmatrix.trans: matrices (n-tile 2hf + (lane >> 4), k-half (lane >> 3) & 1), a lane's row is token
// 16 ks + 8 k-half + (lane & 7), its 16-B piece state rows 8 n-tile..  96 mma.
template <int NP = 3>
__device__ __forceinline__ void gtc_update_state(float (&s)[16][4], const float (&dd)[4][4], const uint16_t* kh,
                                                 const uint16_t* kl) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * ((lane >> 3) & 1) + (lane & 7)) * GTC_KP + 8 * (lane >> 4);
#pragma unroll
    for (int ks = 0; ks < GCH / 16; ++ks) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(dd[2 * ks][0], dd[2 * ks][1], ah[0], al[0]);
        gtc_split_f16x2(dd[2 * ks][2], dd[2 * ks][3], ah[1], al[1]);
        gtc_split_f16x2(dd[2 * ks + 1][0], dd[2 * ks + 1][1], ah[2], al[2]);
        gtc_split_f16x2(dd[2 * ks + 1][2], dd[2 * ks + 1][3], ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < S / 16; ++hf) {
            const int o = off + 16 * ks * GTC_KP + 16 * hf;
            uint32_t bh[4], bl[4] = {0u, 0u, 0u, 0u};
            gtc_ldsm_x4_trans(bh, kh + o);
            if constexpr (NP == 3) gtc_ldsm_x4_trans(bl, kl + o);
            gtc_mmaN<NP>(s[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mmaN<NP>(s[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}

// ---- the prep: gdn_chunk_prep_kernel (kernels.cu) for the tensor-core scan, a block per (chunk, key head): the same
// A, solve, T and P, but the Gram products k_t.k_i and q_t.k_i on the tensor cores (f16x3 from the k and q hi / lo
// images x 2^8, which the block builds first in shared memory straight from the float4 registers the rows arrive in,
// rows past the chunk's end zero: 48 mma a warp against ~1,150 LDS a thread for the FFMA form, which was LDS-bound at
// 233 us a 2048-token super-block), and it writes what the scan stages by cp.async with no conversion of its own: the
// same k / q images, P as f16 hi / lo images x 2^8 (the split the scan did, element by element) and the token scalars
// (beta, exp(gamma), exp(gL - gamma), gamma) as a float4 a token, T as f16 hi / lo images x 2^et, et from the warp's
// largest |T| (2^-et beside them in te).  The Gram entries are within 3 x 2^-22 of exact (the scan's bound), so T and
// P differ from gdn_chunk_prep_kernel's by that.
constexpr size_t kTcPrepSmem = (size_t) 4 * GCH * GTC_KP * 2 + 2 * GCH * GTP * 4;   // the images, then k_t.k_i and q_t.k_i
static_assert(VPK * GCH * GTP * 4 <= 4 * GCH * GTC_KP * 2, "gdn_tc_prep_kernel: A fits where the images were");
__global__ void __launch_bounds__(256) gdn_tc_prep_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                          const float* __restrict__ beta, uint16_t* __restrict__ ti,
                                                          float* __restrict__ te, float4* __restrict__ sc,
                                                          uint16_t* __restrict__ kq, uint16_t* __restrict__ pi, int64_t t0,
                                                          int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    uint16_t* im = reinterpret_cast<uint16_t*>(gsm);   // [4][GCH][GTC_KP]  k hi, k lo, q hi, q lo (x 2^8)
    float* skk = gsm + 4 * GCH * GTC_KP / 2;           // [GCH][GTP]  k_t.k_i
    float* sqk = skk + GCH * GTP;                      // [GCH][GTP]  q_t.k_i
    float* sa = gsm;                                   // [VPK][GCH][GTP]  A, then T below the diagonal (over the
                                                       // images: read by then)
    const int qh = blockIdx.y, ch = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t tb = t0 + (int64_t) ch * GCH;
    const int n = (int) min((int64_t) GCH, T - tb);
    // after the Gram: warps j and VPK + j take value head j (lane i: token i), the first A, the solve and T, the
    // second P and the token scalars; their beta and gate (past the end: gate 0, beta 0, no effect on S) are loaded
    // now so that they are in by then
    const int j = warp % VPK, vh = qh + j * HK, i = lane;
    const bool pv = warp < 2 * VPK && i < n;
    const float b = pv ? beta[(tb + i) * HV + vh] : 0.0f;
    float gam = pv ? gate[(tb + i) * HV + vh] : 0.0f;
    {   // k and q as float4, every load issued before the first store; the images to shared and to global
        constexpr int NP = GCH * S / 4 / 256;   // float4 a thread, each of k and q
        float4 kv[NP], qv[NP];
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            kv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + HK * S + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
            qv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        uint16_t* img = kq + ((size_t) ch * HK + qh) * 4 * GCH * S;
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            const int o = t * GTC_KP + r;
            gtc_split_store4(im + o, im + GCH * GTC_KP + o, kv[m], 256.0f);
            gtc_split_store4(im + 2 * GCH * GTC_KP + o, im + 3 * GCH * GTC_KP + o, qv[m], 256.0f);
            gtc_split_store4(img + t * S + r, img + GCH * S + t * S + r, kv[m], 256.0f);
            gtc_split_store4(img + 2 * GCH * S + t * S + r, img + 3 * GCH * S + t * S + r, qv[m], 256.0f);
        }
    }
    __syncthreads();
    {   // the Gram products: warp w takes product w >> 2 (k_t.k_i, q_t.k_i), rows 16 ((w >> 1) & 1).. (the A
        // tile, by ldmatrix as a row-major [t][r] 16 x 16), columns 16 (w & 1).. (two n-tiles: B = k^T, stored
        // [n][k] = the k rows, by ldmatrix as the scan's gtc_prod_state); the accumulators come out x 2^16
        const int pr = warp >> 2, mt = (warp >> 1) & 1, np = warp & 1;
        const uint16_t* ah_ = im + 2 * pr * GCH * GTC_KP + (16 * mt + (lane & 7) + 8 * ((lane >> 3) & 1)) * GTC_KP + 8 * (lane >> 4);
        const uint16_t* bh_ = im + (16 * np + 8 * (lane >> 4) + (lane & 7)) * GTC_KP + 8 * ((lane >> 3) & 1);
        float c[2][4] = {};
#pragma unroll
        for (int kk = 0; kk < S / 16; ++kk) {
            uint32_t ah[4], al[4], bh[4], bl[4];
            gtc_ldsm_x4(ah, ah_ + 16 * kk);
            gtc_ldsm_x4(al, ah_ + GCH * GTC_KP + 16 * kk);
            gtc_ldsm_x4(bh, bh_ + 16 * kk);
            gtc_ldsm_x4(bl, bh_ + GCH * GTC_KP + 16 * kk);
            gtc_mma3_f16(c[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(c[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
        float* g = pr ? sqk : skk;
        const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int t = 16 * mt + gid + 8 * (q >> 1), i = 16 * np + 8 * j + 2 * tig + (q & 1);
                g[t * GTP + i] = i <= t ? c[j][q] * gtc_pow2(-16) : 0.0f;
            }
    }
    __syncthreads();
    if (warp >= 2 * VPK) return;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, gam, o);
        if (lane >= o) gam += u;
    }
    if (warp >= VPK) {   // P = exp(gamma_t - gamma_i) q_t.k_i (i <= t) x 2^8 as f16 hi / lo; the scalars
        uint16_t* Ph = pi + ((int64_t) ch * HV + vh) * 2 * GCH * GCH;
        uint16_t* Pl = Ph + GCH * GCH;
#pragma unroll
        for (int t = 0; t < GCH; ++t) {
            const float gt = __shfl_sync(0xffffffffu, gam, t);
            const float p = (i <= t ? expf(gt - gam) * sqk[t * GTP + i] : 0.0f) * 256.0f;
            const __half hi = __float2half_rn(p);
            Ph[t * GCH + i] = __half_as_ushort(hi);
            Pl[t * GCH + i] = __half_as_ushort(__float2half_rn(p - __half2float(hi)));
        }
        const float gL = __shfl_sync(0xffffffffu, gam, GCH - 1);
        sc[((int64_t) ch * HV + vh) * GCH + i] = make_float4(b, expf(gam), expf(gL - gam), gam);
        return;
    }
    float* a = sa + j * GCH * GTP;
#pragma unroll
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t), bt = __shfl_sync(0xffffffffu, b, t);
        a[t * GTP + i] = i < t ? bt * expf(gt - gam) * skk[t * GTP + i] : (i == t ? 1.0f : 0.0f);
    }
    __syncwarp();
    // T = (I + A)^-1 by forward substitution, row by row: T[t][i] = -sum_{k < t} A[t][k] T[k][i] over the finished
    // rows (diagonal 1, zeros above it: the terms k < i add an exact 0, so the same bits as summing from k = i).
    // Lane i keeps its column T[.][i] in registers, so a row needs only A's row t (a broadcast load a term) and A
    // is never written: no store and no __syncwarp a row
    float tv[GCH];
    tv[0] = i == 0 ? 1.0f : 0.0f;
#pragma unroll
    for (int t = 1; t < GCH; ++t) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < t; ++k) v = fmaf(-a[t * GTP + k], tv[k], v);
        tv[t] = i < t ? v : (i == t ? 1.0f : 0.0f);
    }
    // T's images x 2^et (its largest entry in [2^14, 2^15), as the scan scales its own operands)
    float mt4[4] = {};
#pragma unroll
    for (int t = 0; t < GCH; ++t) mt4[t & 3] = fmaxf(mt4[t & 3], fabsf(tv[t]));
    const int et = gtc_scale_exp(gtc_warp_max(fmaxf(fmaxf(mt4[0], mt4[1]), fmaxf(mt4[2], mt4[3]))));
    const float sT = gtc_pow2(et);
    uint16_t* Th = ti + ((int64_t) ch * HV + vh) * 2 * GCH * GCH;
    uint16_t* Tl = Th + GCH * GCH;
#pragma unroll
    for (int t = 0; t < GCH; ++t) {
        const float x = tv[t] * sT;
        const __half hi = __float2half_rn(x);
        Th[t * GCH + i] = __half_as_ushort(hi);
        Tl[t * GCH + i] = __half_as_ushort(__float2half_rn(x - __half2float(hi)));
    }
    if (i == 0) te[(int64_t) ch * HV + vh] = gtc_pow2(-et);
}

// ---- the scan: a block per value head (48 blocks, one wave on 48 of the CMP 170HX's 70 SMs), 8 consumer warps of 16
// value columns each walking every chunk of the super-block with their state slice in registers (written back at the
// end as gdn_chunk_scan_kernel does), the prep's scratch from gdn_tc_prep_kernel.  Everything a chunk needs arrives by
// cp.async, issued for the next chunk right after the one __syncthreads a chunk (which publishes the buffer just
// landed and ends every read of the other one): the k / q and P f16 images, T and the token scalars as the prep
// wrote them, v from h (its rows past the chunk's end zero-filled).  The 256 threads do no conversion of their own.
// Earlier forms, measured on this card: the scan splitting k / q itself after the state update (0.282 us a token;
// ~300 split instructions a warp a chunk and the staging exposed at the chunk's end); two stager warps with named
// barriers (6 % slower: registers are allocated per SM sub-partition, 16K each, and a block of more than 8 warps puts
// 3 on a sub-partition, which caps the kernel at 168 registers a thread; the consumers need about 200 and spilled.
// 8 warps, 2 a sub-partition, may have 255).
// NORM (step 5: the output norm fused, the FP16 output written here) is not implemented yet; oc_out takes the FP32
// output as gdn_chunk_scan_kernel writes it.
template <bool NORM, bool SPLIT = false, int NP = 3, int NTL = 1>
__global__ void __launch_bounds__(GTC_CW * 32, 1)
gdn_tc_scan_kernel(float* __restrict__ state, const float* __restrict__ h, const uint16_t* __restrict__ tig_,
                   const float* __restrict__ teg, const float4* __restrict__ scg, const uint16_t* __restrict__ kqg,
                   const uint16_t* __restrict__ pig, float* __restrict__ oc_out, int64_t t0, int nch, int64_t T,
                   const float* __restrict__ zg = nullptr, const float* __restrict__ gamma = nullptr, float eps = 0.0f,
                   uint16_t* __restrict__ y16 = nullptr, int64_t ld16 = 0) {
    // NORM (SPLIT only): the output norm of gdn_out_norm_kernel fused: the scan writes the FP16 y16 = o rsqrt(mean(o^2) + eps) gamma
    // sigmoid(z) itself (the sums of squares across the block's 8 warps through shared memory), and not the FP32 o: one pass less
    // over the 24 KB-a-token output.  Another summation order than the kernel's 4 partials (not bitwise).
    static_assert(!NORM || SPLIT, "gdn_tc_scan_kernel: the fused output norm is implemented for the split plan");
    // NTL value tiles a head (a launch of HV x NTL blocks of GTC_CW / NTL warps): block (vh, tile) owns the state columns
    // [tile S / NTL, (tile + 1) S / NTL), so a card with more than 48 SMs keeps more of them busy.  The k / q / T / P images are staged
    // by every tile of the head; v only for its columns.  The fused output norm needs the whole head in one block: NORM => NTL == 1.
    static_assert(!NORM || NTL == 1, "gdn_tc_scan_kernel: the fused output norm needs a whole head in a block");
    static_assert(GTC_CW % NTL == 0, "gdn_tc_scan_kernel: tiles of whole warps");
    constexpr int NT = (GTC_CW / NTL) * 32;
    extern __shared__ __align__(16) unsigned char gtc_sm[];
    uint16_t* kh = reinterpret_cast<uint16_t*>(gtc_sm);                 // [2][GCH][GTC_KP]  f16 of k x 2^8 (hi)
    uint16_t* kl = kh + 2 * GCH * GTC_KP;                               // [2][GCH][GTC_KP]  what hi left out (lo)
    uint16_t* qh = kl + 2 * GCH * GTC_KP;                               // q likewise
    uint16_t* ql = qh + 2 * GCH * GTC_KP;
    constexpr int NV = SPLIT ? 1 : 2;                                   // buffers of v, T and P
    float* sv = reinterpret_cast<float*>(ql + 2 * GCH * GTC_KP);        // [NV][GCH][GTC_VP]  v
    uint16_t* th = reinterpret_cast<uint16_t*>(sv + NV * GCH * GTC_VP); // [NV][GCH][GTC_PP]  T x 2^et hi / lo
    uint16_t* tl = th + NV * GCH * GTC_PP;
    uint16_t* ph = tl + NV * GCH * GTC_PP;                              // [NV][GCH][GTC_PP]  P x 2^8 hi / lo
    uint16_t* pl = ph + NV * GCH * GTC_PP;
    float4* sc = reinterpret_cast<float4*>(pl + NV * GCH * GTC_PP);     // [2][GCH]  (beta, exp(gamma), exp(gL - gamma), gamma)
    float* ste = reinterpret_cast<float*>(sc + 2 * GCH);                // [2]  2^-et
    float* rss = ste + 2;                                               // NORM: [GTC_CW][GCH]  partial sums of squares
    const int vh = blockIdx.x / NTL, qhd = vh % HK, cb = (blockIdx.x % NTL) * (S / NTL);   // the tile's first column
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const size_t rs = (size_t) HV * S;
    // chunk ch into buffer b, this thread's 16-B pieces (one cp.async group): the four k / q images (4 x GCH rows of
    // 16 pieces: 8 a thread), v (4 a thread), the two T images (1), the two P images (1), the scalars (warp 0, 1),
    // T's scale (thread 0, 4 B)
    // parts: bit 0 = the k / q images, the scalars and T's scale (buffer b); bit 1 = v, T and P (buffer bv: b, or 0 in SPLIT)
    auto stage = [&](int ch, int b, int parts) {
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        const int bv = SPLIT ? 0 : b;
        if (parts & 1) {
            const uint16_t* img = kqg + ((size_t) ch * HK + qhd) * 4 * GCH * S;
#pragma unroll
            for (int m = 0; m < 4 * GCH * S / 8 / NT; ++m) {
                const int pc = tid + NT * m, im = pc / (GCH * S / 8), t = (pc % (GCH * S / 8)) / (S / 8), r = (pc % (S / 8)) * 8;
                uint16_t* dst = (im == 0 ? kh : im == 1 ? kl : im == 2 ? qh : ql) + (b * GCH + t) * GTC_KP + r;
                gdn_cp16(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(img + (im * GCH + t) * S + r));
            }
            if (tid < GCH) gdn_cp16(reinterpret_cast<float*>(sc + b * GCH + tid), reinterpret_cast<const float*>(scg + ((int64_t) ch * HV + vh) * GCH + tid));
            if (tid == 0) gdn_cp4(ste + b, teg + (int64_t) ch * HV + vh);
        }
        if (parts & 2) {
#pragma unroll
            for (int m = 0; m < GCH * (S / NTL) / 4 / NT; ++m) {
                const int p = tid + NT * m, t = p / (S / NTL / 4), r = (p % (S / NTL / 4)) * 4;
                float* dst = sv + (bv * GCH + t) * GTC_VP + r;
                if (t < n) gdn_cp16(dst, h + (c_t0 + t) * C + 2 * HK * S + vh * S + cb + r);
                else *reinterpret_cast<float4*>(dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll
            for (int m = 0; m < 2 * GCH * GCH / 8 / NT; ++m) {
                const int pc = tid + NT * m, im = pc / (GCH * GCH / 8), t = (pc % (GCH * GCH / 8)) / (GCH / 8), i = (pc % (GCH / 8)) * 8;
                const int64_t src = ((((int64_t) ch * HV + vh) * 2 + im) * GCH + t) * GCH + i;
                const int dst = (bv * GCH + t) * GTC_PP + i;
                gdn_cp16(reinterpret_cast<float*>((im ? tl : th) + dst), reinterpret_cast<const float*>(tig_ + src));
                gdn_cp16(reinterpret_cast<float*>((im ? pl : ph) + dst), reinterpret_cast<const float*>(pig + src));
            }
        }
        gdn_cp_commit();
    };
    const int c0 = 16 * warp;     // the warp's value columns c0 + gid and c0 + gid + 8 of the tile (v in shared memory)
    const int ca = cb + c0;       // the same in the head (state, output)
    // the state slice as S^T fragments: tile j = rows 8j..8j+7; s[j][0] = S[8j + 2tig][c0 + gid], [1] the row after,
    // [2] column + 8, [3] both (64 scalar loads a lane, 32-B runs, once a launch)
    float s[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float* p = state + (size_t) (8 * j + 2 * tig) * rs + (size_t) vh * S + ca + gid;
        s[j][0] = p[0]; s[j][1] = p[rs]; s[j][2] = p[8]; s[j][3] = p[rs + 8];
    }
    stage(0, 0, 3);
    const float out_scale = rsqrtf((float) S);
    for (int ch = 0; ch < nch; ++ch) {
        const int b = ch & 1;
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        gdn_cp_wait_all();   // this thread's pieces of chunk ch
        __syncthreads();     // every thread's: chunk ch's buffer complete; every read of the other one done
        if (ch + 1 < nch) stage(ch + 1, b ^ 1, SPLIT ? 1 : 3);
        const uint16_t* bkh = kh + b * GCH * GTC_KP;
        const uint16_t* bkl = kl + b * GCH * GTC_KP;
        const uint16_t* bqh = qh + b * GCH * GTC_KP;
        const uint16_t* bql = ql + b * GCH * GTC_KP;
        const int bv = SPLIT ? 0 : b;
        const uint16_t* bth = th + bv * GCH * GTC_PP;
        const uint16_t* btl = tl + bv * GCH * GTC_PP;
        const uint16_t* bph = ph + bv * GCH * GTC_PP;
        const uint16_t* bpl = pl + bv * GCH * GTC_PP;
        const float* bsv = sv + bv * GCH * GTC_VP;
        const float4* bsc = sc + b * GCH;
        // (a) X^T = S0^T K^T and QS^T = S0^T Q^T, the state x 2^es (its largest entry in [2^14, 2^15)).  The lane's
        // maximum over eight chains (nvcc keeps the order written: one chain was 64 dependent FMNMX before the first mma)
        float m8[8] = {};
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) m8[(2 * j + (q >> 1)) & 7] = fmaxf(m8[(2 * j + (q >> 1)) & 7], fabsf(s[j][q]));
        const float m = fmaxf(fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])), fmaxf(fmaxf(m8[4], m8[5]), fmaxf(m8[6], m8[7])));
        const int es = gtc_scale_exp(gtc_warp_max(m));
        float xt[4][4] = {}, qs[4][4] = {};
        gtc_prod_state<(NP == 1 ? 1 : 3), (NP == 3 ? 3 : 1)>(s, gtc_pow2(es), bkh, bkl, bqh, bql, xt, qs);
        // (b) Y^T = beta_i (V^T - e'_i X^T), e'_i = exp(gamma_i) 2^-(8 + es): lane (gid, tig) holds tokens
        // 8nt + 2tig and + 1, columns c0 + gid and + 8
        const float eS = gtc_pow2(-(8 + es));
        float yt[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float4 s0 = gtc_lds128(bsc + 8 * nt + 2 * tig), s1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1);
            const float* v = bsv + (8 * nt + 2 * tig) * GTC_VP + c0 + gid;
            const float e0 = s0.y * eS, e1 = s1.y * eS;
            yt[nt][0] = s0.x * fmaf(-e0, xt[nt][0], v[0]);
            yt[nt][1] = s1.x * fmaf(-e1, xt[nt][1], v[GTC_VP]);
            yt[nt][2] = s0.x * fmaf(-e0, xt[nt][2], v[8]);
            yt[nt][3] = s1.x * fmaf(-e1, xt[nt][3], v[GTC_VP + 8]);
        }
        // (c) D^T = Y^T T^T on f16x3: Y^T x 2^ey (its largest entry in [2^14, 2^15)), T's images x 2^et from the prep;
        // the accumulator comes out x 2^(ey + et) (T above its diagonal skipped)
        float my4[4] = {};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) my4[q] = fmaxf(my4[q], fabsf(yt[nt][q]));
        const int ey = gtc_scale_exp(gtc_warp_max(fmaxf(fmaxf(my4[0], my4[1]), fmaxf(my4[2], my4[3]))));
        float dt[4][4] = {};
        gtc_prod_dt<true, (NP == 1 ? 1 : 3)>(yt, gtc_pow2(ey), bth, btl, dt);
        const float sYT = gtc_pow2(-ey) * ste[b];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) dt[nt][q] *= sYT;
        // (d) o^T = e_t QS^T + D^T P^T: the accumulator starts as 2^(ed + 8) e_t QS^T (QS^T came out x 2^(8 + es)),
        // D^T goes in x 2^ed, P x 2^8; the output is the accumulator x 2^-(ed + 8), then rsqrt(S) as gdn_chunk_scan_kernel
        float md4[4] = {};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) md4[q] = fmaxf(md4[q], fabsf(dt[nt][q]));
        const int ed = gtc_scale_exp(gtc_warp_max(fmaxf(fmaxf(md4[0], md4[1]), fmaxf(md4[2], md4[3]))));
        const float sD = gtc_pow2(ed), sO = gtc_pow2(ed + 8), sOi = gtc_pow2(-(ed + 8)), eO = gtc_pow2(ed - es);
        float ot[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float e0 = gtc_lds128(bsc + 8 * nt + 2 * tig).y * eO, e1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1).y * eO;
            ot[nt][0] = qs[nt][0] * e0; ot[nt][1] = qs[nt][1] * e1; ot[nt][2] = qs[nt][2] * e0; ot[nt][3] = qs[nt][3] * e1;
        }
        gtc_prod_dt<true, (NP == 3 ? 3 : 1)>(dt, sD, bph, bpl, ot);
        float zr[4][4] = {}, gm2[2] = {};
        if constexpr (NORM) {
            // the z values and gamma of this lane's elements are loaded now, so that their latency is spent in the barrier below
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int t = 8 * nt + 2 * tig + (q & 1);
                    if (t < n) zr[nt][q] = zg[(c_t0 + t) * rs + (size_t) vh * S + c0 + gid + 8 * (q >> 1)];
                }
            gm2[0] = gamma[c0 + gid]; gm2[1] = gamma[c0 + gid + 8];
            // this lane's two columns of each of its 8 tokens -> partial sums over the warp's 16 columns (the 8 gid lanes), 8 warps in rss
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int q = 0; q < 4; ++q) ot[nt][q] = (ot[nt][q] * sOi) * out_scale;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float ps = ot[nt][e] * ot[nt][e] + ot[nt][e + 2] * ot[nt][e + 2];
                    ps += __shfl_xor_sync(0xffffffffu, ps, 4);
                    ps += __shfl_xor_sync(0xffffffffu, ps, 8);
                    ps += __shfl_xor_sync(0xffffffffu, ps, 16);
                    if (gid == 0) rss[warp * GCH + 8 * nt + 2 * tig + e] = ps;
                }
        } else {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int t = 8 * nt + 2 * tig + (q & 1);
                    if (t < n) oc_out[(c_t0 + t) * rs + (size_t) vh * S + ca + gid + 8 * (q >> 1)] = (ot[nt][q] * sOi) * out_scale;
                }
        }
        if (SPLIT) {   // v, T and P of this chunk are dead: every warp is past (d); the next chunk's land during (e)
            __syncthreads();
            if (ch + 1 < nch) stage(ch + 1, 0, 2);
        }
        if constexpr (NORM) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int t = 8 * nt + 2 * tig + e;
                    if (t >= n) continue;
                    float ss = 0.0f;
#pragma unroll
                    for (int w = 0; w < GTC_CW; ++w) ss += rss[w * GCH + t];
                    const float rn = rsqrtf(ss / (float) S + eps);
#pragma unroll
                    for (int hc = 0; hc < 2; ++hc) {
                        const int col = c0 + gid + 8 * hc;
                        const float zz = zr[nt][e + 2 * hc];
                        const float v = ot[nt][e + 2 * hc] * rn * gm2[hc] * (1.0f / (1.0f + __expf(-zz)));
                        y16[(c_t0 + t) * ld16 + (size_t) vh * S + col] = __half_as_ushort(__float2half_rn(v));
                    }
                }
        }
        // (e) S^T = eL S0^T + Dd^T K, Dd^T = f_i D^T: the accumulator x (eL 2^(ed + 8)) before (one rounding, as the
        // s *= gL of gdn_chunk_scan_kernel), Dd^T in x 2^ed, K x 2^8, the accumulator x 2^-(ed + 8) after
        const float sPre = gtc_lds128(bsc + GCH - 1).y * sO;   // exp(gL)
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) s[j][q] *= sPre;
        float dd[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float f0 = gtc_lds128(bsc + 8 * nt + 2 * tig).z * sD, f1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1).z * sD;
            dd[nt][0] = dt[nt][0] * f0; dd[nt][1] = dt[nt][1] * f1; dd[nt][2] = dt[nt][2] * f0; dd[nt][3] = dt[nt][3] * f1;
        }
        gtc_update_state<(NP == 1 ? 1 : 3)>(s, dd, bkh, bkl);
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) s[j][q] *= sOi;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float* p = state + (size_t) (8 * j + 2 * tig) * rs + (size_t) vh * S + ca + gid;
        p[0] = s[j][0]; p[rs] = s[j][1]; p[8] = s[j][2]; p[rs + 8] = s[j][3];
    }
}
#endif   // !__HIPCC__
