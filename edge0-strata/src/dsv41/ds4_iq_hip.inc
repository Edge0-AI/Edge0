// Strata. The prompt sets of ds4_iq.cu on the gfx12 (RDNA4, R9700 = gfx1201) matrix cores: v_wmma_f32_16x16x16_f16, F16 operands,
// F32 accumulation.  Included by ds4_iq.cu inside its anonymous namespace (it uses its Ty, decode<>, load_tab<>, swiglu, SetArgs).
//
// The idea is the CUDA prompt tiles' (set_h16ws_kernel on f-dsv41-cuk-int2): a block-tile of the expert's rows x weight rows, K in
// steps of 64; each i-quant block is decoded ONCE into shared memory as F16 (code * the sub-block's scale: the product is exact up
// to the F16 rounding of the scale and the product) and then shared by every row of the tile; x as F16 rows next to it, both
// double-buffered, one barrier a step.  The F32 FMA tile of today (set_kernel) decodes the weights once per 64 rows and does
// scalar FMAs.  Default ON for gfx12, STRATA_DS4_HIPK=0 switches it off (gfx12 only, the types below, experts of more than 16 rows).
#pragma once
#if defined(STRATA_USE_HIP)

namespace hipk {
// test hooks: force_mode -1 = the default rules, 0 = never, 1 = on wherever the device has the WMMA tiles (gfx11 / gfx12); launches counts the tiles taken
inline std::atomic<int>& force_mode() { static std::atomic<int> f{-1}; return f; }
inline std::atomic<long long>& launches() { static std::atomic<long long> n{0}; return n; }
constexpr int kStH = 72;   // halves a shared row: 64 + 8 (the pad spreads the banks; rows stay 16-byte aligned)
using hv2 = _Float16 __attribute__((ext_vector_type(2)));
using hv8 = _Float16 __attribute__((ext_vector_type(8)));
using fv8 = float __attribute__((ext_vector_type(8)));

template <int T>
constexpr bool kTypes = T == kIQ2_XXS || T == kIQ2_XS || T == kIQ2_S || T == kIQ3_XXS || T == kIQ3_S || T == kIQ4_XS ||
                        T == kIQ1_S || T == kIQ1_M || T == kIQ4_NL;

__device__ __forceinline__ fv8 wmma16(hv8 a, hv8 b, fv8 c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    return c;
#endif
}
using hv16 = _Float16 __attribute__((ext_vector_type(16)));
// RDNA3 / RDNA3.5 (gfx11): the same v_wmma_f32_16x16x16_f16 on wave32, but a lane holds ALL 16 k of its row / column (the two
// half-waves the same values) and the D rows a lane holds are 2 q + hi (gfx12: 8 hi + q).  DS4_HIPK_G11 is set in the device pass only.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1152__)
#define DS4_HIPK_G11 1
#endif
__device__ __forceinline__ fv8 wmma16g11(hv16 a, hv16 b, fv8 c) {
#if defined(DS4_HIPK_G11)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    return c;
#endif
}
__device__ __forceinline__ uint32_t as_u32(hv2 v) { return __builtin_bit_cast(uint32_t, v); }
__device__ __forceinline__ hv2 as_hv2(uint32_t v) { return __builtin_bit_cast(hv2, v); }
// 4 int8 codes (a word) -> 4 halves times the scale: 1024 + (code + 128) is exact in a half's mantissa, less 1152
__device__ __forceinline__ uint2 codes4(int w, hv2 sc) {
    const uint32_t u = (uint32_t) w ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x64646464u, 0x5140), hi = __byte_perm(u, 0x64646464u, 0x5342);
    const hv2 m = {(_Float16) 1152.0f, (_Float16) 1152.0f};
    const hv2 a = (as_hv2(lo) - m) * sc, b = (as_hv2(hi) - m) * sc;
    return make_uint2(as_u32(a), as_u32(b));
}

// BM rows x BN weight rows (a matrix: gate and up: two) per block, 8 waves WM x WN, each wave (BM/WM) x (BN/WN)
template <int T, bool GU, int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(256) set_kernel(const SetArgs a, int dbg) {
    static_assert(WM * WN == 8, "8 waves");
    constexpr int NM = GU ? 2 : 1, MT = BM / WM / 16, NT = BN / WN / 16;
    static_assert(MT >= 1 && NT >= 1 && MT * WM * 16 == BM && NT * WN * 16 == BN, "wave tiling");
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ int srow[BM];
    __shared__ __align__(16) _Float16 xs[2][BM][kStH];
    __shared__ __align__(16) _Float16 ws[2][NM][BN][kStH];
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi_r = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi_r || hi_r - lo <= a.skip_le) return;   // (uniform)
    load_tab<T>(tab);
    const int nr = min(BM, hi_r - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GU ? a.n_embd : a.n_ff;
    const size_t rb = row_bytes_t<T>(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    if (tid < BM) srow[tid] = GU ? a.idx[r0 + min(tid, nr - 1)] : r0 + min(tid, nr - 1);
    __syncthreads();
    const int nst = K / 64;

    // a step's operands are fetched and decoded into registers (the global loads and the codebook's lookups) while the matrix
    // cores work on the step before, and written to shared memory after them
    constexpr int NI = (NM * BN * 2 + 255) / 256;                // weight items a thread
    constexpr int NX = GU ? (BM * 16 + 255) / 256 : (BM * 8 + 255) / 256;   // row chunks a thread
    struct Regs {
        int q[NI][8];
        _Float16 sc0[NI], sc1[NI];
        float4 xv[GU ? NX : 1];
        uint4 xu[GU ? 1 : NX];
    };
    auto fetch = [&](int s, Regs& R) {
#pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            const int it = tid + ii * 256;
            if (it < NM * BN * 2) {
                const int kbl = it & 1, col = (it >> 1) % BN, mat = it / (2 * BN);
                Sub sb;
                if (dbg & 2) { for (int w = 0; w < 8; ++w) sb.q[w] = tid + w; sb.d = 1.0f; sb.s0 = sb.s1 = 1; } else
                decode<T>(blob + a.tr[GU ? mat : 2] + (size_t) (c0 + col) * rb, s * 2 + kbl, tab, sb);
#pragma unroll
                for (int w = 0; w < 8; ++w) R.q[ii][w] = sb.q[w];
                R.sc0[ii] = (_Float16) (sb.d * (float) sb.s0);
                R.sc1[ii] = (_Float16) (sb.d * (float) sb.s1);
            }
        }
#pragma unroll
        for (int xi = 0; xi < NX; ++xi) {
            const int c = tid + xi * 256;
            if constexpr (GU) {
                if (c < BM * 16) {
                    const int row = c >> 4, ch = c & 15;
                    R.xv[xi] = *(const float4*) (a.xf + (size_t) srow[row] * a.ldx + (size_t) s * 64 + ch * 4);
                }
            } else {
                if (c < BM * 8) {
                    const int row = c >> 3, ch = c & 7;
                    R.xu[xi] = *(const uint4*) (a.xh + (size_t) srow[row] * a.ldx + (size_t) s * 64 + ch * 8);
                }
            }
        }
    };
    auto store = [&](const Regs& R, int b) {
#pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            const int it = tid + ii * 256;
            if (it < NM * BN * 2) {
                const int kbl = it & 1, col = (it >> 1) % BN, mat = it / (2 * BN);
                const hv2 h0 = {R.sc0[ii], R.sc0[ii]}, h1 = {R.sc1[ii], R.sc1[ii]};
                uint2* dst = (uint2*) &ws[b][mat][col][kbl * 32];   // (values 4 w .. 4 w + 3 of the half at dst[w], 16 + 4 w .. at dst[4 + w])
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    dst[w] = codes4(R.q[ii][w], h0);
                    dst[4 + w] = codes4(R.q[ii][4 + w], h1);
                }
            }
        }
#pragma unroll
        for (int xi = 0; xi < NX; ++xi) {
            const int c = tid + xi * 256;
            if constexpr (GU) {
                if (c < BM * 16) {
                    const int row = c >> 4, ch = c & 15;
                    const float4 v = R.xv[xi];
                    const hv2 p = {(_Float16) v.x, (_Float16) v.y}, q = {(_Float16) v.z, (_Float16) v.w};
                    *(uint2*) &xs[b][row][ch * 4] = make_uint2(as_u32(p), as_u32(q));
                }
            } else {
                if (c < BM * 8) {
                    const int row = c >> 3, ch = c & 7;
                    *(uint4*) &xs[b][row][ch * 8] = R.xu[xi];
                }
            }
        }
    };

    const int wm = warp / WN, wn = warp % WN;
    fv8 acc[NM][MT][NT];
#pragma unroll
    for (int m = 0; m < NM; ++m)
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) acc[m][i][j] = fv8{0, 0, 0, 0, 0, 0, 0, 0};

    // read-ahead: the first and last byte of the weights' block and a word of the rows' line of a step PF steps on, summed into a
    // value that is never used (the loads only warm the caches; their latency is not waited for until the end)
    int touched = 0;
    auto touch = [&](int s) {
        if (dbg & 4) return;
#pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            const int it = tid + ii * 256;
            if (it < NM * BN * 2) {
                const int col = (it >> 1) % BN, mat = it / (2 * BN);
                const uint8_t* p = blob + a.tr[GU ? mat : 2] + (size_t) (c0 + col) * rb + (size_t) ((s * 2) >> 3) * Ty<T>::B;
                touched += p[0] + p[Ty<T>::B - 1];
            }
        }
        if constexpr (GU) {
            for (int c = tid; c < BM * 2; c += 256) touched += (int) a.xf[(size_t) srow[c >> 1] * a.ldx + (size_t) s * 64 + (c & 1) * 32];
        } else {
            for (int c = tid; c < BM * 2; c += 256) touched += (int) a.xh[(size_t) srow[c >> 1] * a.ldx + (size_t) s * 64 + (c & 1) * 32];
        }
    };
    constexpr int PF = 4;
    for (int s = 1; s <= PF && s < nst; ++s) touch(s + 1 < nst ? s + 1 : s);
    {
        Regs R0;
        fetch(0, R0);
        store(R0, 0);
    }
    __syncthreads();
    for (int s = 0; s < nst; ++s) {
        const int b = s & 1;
        Regs R;
        if (s + 1 < nst) fetch(s + 1, R);
        if (s + 1 + PF < nst) touch(s + 1 + PF);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            hv8 af[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) af[i] = *(const hv8*) &xs[b][wm * (16 * MT) + i * 16 + l16][kk * 16 + hi * 8];
#pragma unroll
            for (int m = 0; m < NM; ++m)
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const hv8 bf = *(const hv8*) &ws[b][m][wn * (16 * NT) + j * 16 + l16][kk * 16 + hi * 8];
#pragma unroll
                    for (int i = 0; i < MT; ++i)
                        if (!(dbg & 1) && wm * (16 * MT) + i * 16 < nr) acc[m][i][j] = wmma16(af[i], bf, acc[m][i][j]);   // (a tile past the rows: skipped, uniform)
                }
        }
        if (s + 1 < nst) store(R, b ^ 1);
        __syncthreads();
    }
    if (touched == 0x7fffffff && tid == 12345) a.y[0] = 0.0f;
    // D[row = 8 hi + i][col = l16] of every tile
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int row = wm * (16 * MT) + i * 16 + 8 * hi + q;
            if (row >= nr) continue;
            const size_t lr = (size_t) (r0 + row);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int col = c0 + wn * (16 * NT) + j * 16 + l16;
                if (GU) a.h[lr * a.ldo + col] = __half_as_ushort(__float2half_rn(swiglu(acc[0][i][j][q], acc[1][i][j][q], a.limit)));
                else a.y[lr * a.ldo + col] = acc[0][i][j][q];
            }
        }
}

#include "ds4_iq_hip_r.inc"

inline bool ok_device() {   // gfx12 unless STRATA_DS4_HIPK=0
    static std::atomic<int> state[16];
    static std::atomic<int> capable[16];   // 1: gfx11 / gfx12
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return false;
    if (const int f = force_mode().load(std::memory_order_relaxed); f >= 0) {
        if (f == 0) return false;
        int c = capable[dev].load(std::memory_order_relaxed);
        if (c == 0) {
            cudaDeviceProp pr{};
            const bool g = cudaGetDeviceProperties(&pr, dev) == cudaSuccess && (std::strncmp(pr.gcnArchName, "gfx12", 5) == 0 || std::strncmp(pr.gcnArchName, "gfx11", 5) == 0);
            c = g ? 1 : 2;
            capable[dev].store(c, std::memory_order_relaxed);
        }
        return c == 1;
    }
    int v = state[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        cudaDeviceProp prop{};
        const bool ok = cudaGetDeviceProperties(&prop, dev) == cudaSuccess;
        const bool gfx12 = ok && std::strncmp(prop.gcnArchName, "gfx12", 5) == 0;
        const bool gfx11 = ok && std::strncmp(prop.gcnArchName, "gfx11", 5) == 0;
        const char* e = std::getenv("STRATA_DS4_HIPK");
        v = gfx12 && !(e != nullptr && e[0] == '0') ? 1 : 2;   // (on by default on gfx12; STRATA_DS4_HIPK=0 off)
        // gfx11: the register-weights tiles only.  On by default on gfx1151 (Strix Halo: prompts +22..29% at the same KL, measured), opt-in
        // (STRATA_DS4_HIPK=1) on the other gfx11 parts, which were not measured; STRATA_DS4_HIPK=0 turns it off everywhere
        if (gfx11) {
            const bool halo = std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
            v = (e != nullptr ? e[0] == '1' : halo) ? 3 : 2;
        }
        state[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1 || v == 3;
}
inline bool is_g11() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    static std::atomic<int> g[16];
    if (dev < 0 || dev >= 16) return false;
    int v = g[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        v = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx11", 5) == 0 ? 1 : 2;
        g[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

template <int T, bool GU, int BM, int BN, int WM, int WN>
inline bool run(const SetArgs& a, cudaStream_t s) {
    const int N = GU ? a.n_ff : a.n_embd;
    const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM), (unsigned) (N / BN), (unsigned) a.n_exp);
    static const int dbg = std::getenv("HIPK_DBG") ? std::atoi(std::getenv("HIPK_DBG")) : 0;
    set_kernel<T, GU, BM, BN, WM, WN><<<grid, 256, 0, s>>>(a, dbg);
    launches().fetch_add(1, std::memory_order_relaxed);
    return cudaPeekAtLastError() == cudaSuccess;
}
// false: not taken (the caller's tiles do it)
template <int T, bool GU>
inline bool launch(const SetArgs& a, cudaStream_t s) {
    if constexpr (!kTypes<T>) {
        return false;
    } else {
        if (!ok_device() || a.max_rows <= 16) return false;
        const int K = GU ? a.n_embd : a.n_ff, N = GU ? a.n_ff : a.n_embd;
        if (K % 256 != 0 || N % 64 != 0 || a.ldx % 8 != 0 || a.ldo % 2 != 0) return false;
        if (((uintptr_t) (GU ? (const void*) a.xf : (const void*) a.xh) & 15) != 0) return false;
        static const int var = std::getenv("STRATA_DS4_HIPK") ? std::atoi(std::getenv("STRATA_DS4_HIPK")) : 0;
        if (is_g11() && var == 2) return false;   // (the shared-memory weights kernel is gfx12 only)
        if (var != 2) {   // 2: the shared-memory weights kernel
            static const int cfg = std::getenv("HIPK_CFG") ? std::atoi(std::getenv("HIPK_CFG")) : 0;
            if (cfg == 1 && a.max_rows <= 32) return runr<T, GU, 32, 4, 1>(a, s);
            if (cfg == 2 && a.max_rows <= 32) return runr<T, GU, 32, 8, 1>(a, s);
            if (cfg == 3 && a.max_rows <= 32) return runr<T, GU, 32, 2, 2>(a, s);
            if (cfg == 4 && a.max_rows <= 32) return runr<T, GU, 32, 8, 2>(a, s);
            if (cfg == 5 && a.max_rows <= 32) return runr<T, GU, 32, 2, 1>(a, s);
            // (MarLin tile sweep on long sets: a shorter block for the tail rows - one kernel, same arithmetic)
            if (cfg == 6) return runr<T, GU, 64, 8, 1>(a, s);
            if (cfg == 7) return runr<T, GU, 32, 8, 1>(a, s);
            if (cfg == 8) return runr<T, GU, 128, 8, 2>(a, s);
            if (cfg == 9) return runr<T, GU, 64, 4, 2>(a, s);
            // (a wider column block for the down pass only: its K is short and its F16 activation tile is re-read by
            // every column block - 40 of them at BN 128, 20 at BN 256, the weight tile unchanged)
            if (cfg == 10) {
                if constexpr (GU) return runr<T, GU, 128, 8, 1>(a, s);
                else return runr<T, GU, 128, 8, 2>(a, s);
            }
            // (64 rows x 256 columns a block: half the activation re-reads of 128 x 128 at the same accumulator count)
            if (cfg == 11) return runr<T, GU, 64, 8, 2>(a, s);
            if (a.max_rows <= 32) return runr<T, GU, 32, 4, 2>(a, s);
            if (a.max_rows <= 64) return runr<T, GU, 64, 4, 2>(a, s);
            return runr<T, GU, 128, 8, 1>(a, s);
        }
        if (a.max_rows <= 32) return run<T, GU, 32, 64, 2, 4>(a, s);
        if (a.max_rows <= 64) return run<T, GU, 64, 64, 2, 4>(a, s);
        if constexpr (GU) return run<T, GU, 128, 32, 4, 2>(a, s);
        else return run<T, GU, 128, 64, 4, 2>(a, s);
    }
}
}  // namespace hipk

#endif
