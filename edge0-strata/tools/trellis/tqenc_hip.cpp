// tools/trellis/tqenc_hip.cpp - the trellis encoder (tqenc.hpp) on a GPU (HIP, gfx1201 / any wave32 RDNA): TQK4X / TQK5X /
// TQK6 / TQK7 (TQ2_T stays CPU-only: its predecessor groups are strided across threads).
//
// One workgroup (512 threads) runs one Viterbi pass over one 128-weight block: thread t keeps the 64 path costs of the states
// 64 t .. 64 t + 63 in registers.  A step is (a) every thread takes the min/argmin of its predecessor groups (a group
// is 2^K contiguous states, so never crosses a thread for K <= 7) and writes it to LDS plus the argmin to a global
// back-pointer row, (b) after a barrier every thread sets its new costs D[s] = m[s & (G-1)] + c01[p0(s)] + c23[p1(s)]:
// the cost of a state is the sum of the costs of its two hashed codebook points (2048 each), tabulated per step in LDS.
// The same search as the CPU encoder (rotated free pass for the tail-biting start group, constrained pass, fp16 scale
// refinement by least squares), batched over all blocks: every stage is one kernel over the whole batch.
//
// Build: hipcc -O3 --offload-arch=gfx1201 -mavx2 -mfma -mf16c tools/trellis/tqenc_hip.cpp tools/trellis/tqenc.cpp ...
#include "tqenc.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HCHECK(x)                                                                                  \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

constexpr int kSL = 6;        // log2 of the states per thread (64: 512 threads per workgroup)
constexpr int NT = 32768 >> kSL;
constexpr float kInf = 1e30f;

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

// weight index of component j at processing position p (TQK)
__device__ __forceinline__ int widx_d(int p, int j) { return 4 * (31 - p) + j; }

// the computed codebook of TQK5X / TQK4X (src/kernels/cpu/tq_computed.hpp): four fp16 from two LCG words, no table
__device__ __forceinline__ void comp_vals(uint32_t s, float* w) {
    const uint32_t r0 = ((s * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const uint32_t r1 = ((s * 0x9E3779B1u + 0x7F4A7C15u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    w[0] = h2f((uint16_t) (r0 & 0xffffu));
    w[1] = h2f((uint16_t) (r0 >> 16));
    w[2] = h2f((uint16_t) (r1 & 0xffffu));
    w[3] = h2f((uint16_t) (r1 >> 16));
}

__device__ __forceinline__ float comp_cost(uint32_t s, const float* t, const float* h) {
    float w[4];
    comp_vals(s, w);
    const float e0 = t[0] - w[0], e1 = t[1] - w[1], e2 = t[2] - w[2], e3 = t[3] - w[3];
    return h[0] * e0 * e0 + h[1] * e1 * e1 + h[2] * e2 * e2 + h[3] * e3 * e3;
}

struct Pass {
    const float* w;
    const float* h;            // null: ones
    const uint16_t* dcur;      // fp16 scale per block
    const uint8_t* active;
    int64_t nb;
    int rot;
    const int* cin;            // null: free pass, else the tail-biting group per block (<0: skip)
    const float2* lutq;        // 2048 points
    uint16_t* bp_scratch;
    uint16_t* st_out;          // constrained: [nb][32] by position
    float* cost_out;           // constrained: [nb]
    int* c_out;                // free: junction group [nb]
};

template <int K, int SL, bool COMP>
__global__ void __launch_bounds__(32768 >> SL) pass_kernel(Pass P) {
    constexpr int SPT = 1 << SL;          // states per thread
    constexpr int NT = 32768 >> SL;       // threads per workgroup
    constexpr int G = 1 << (15 - K);
    constexpr int SPAN = 1 << K;
    constexpr int GPT = SPT >= SPAN ? SPT / SPAN : 1;   // groups per thread (SPT >= SPAN)
    constexpr int PC = SPT >= SPAN ? 1 : SPAN / SPT;    // threads sharing one group (SPAN > SPT)
    constexpr int NP = 2048 / NT;                       // codebook points per thread
    __shared__ float tab[2][2][2048];
    __shared__ float mbuf[2][G];
    __shared__ float Tt[32][4], Hh[32][4];
    __shared__ float redv[NT / 32];
    __shared__ int redi[NT / 32];

    const int tid = threadIdx.x;
    uint16_t* bp = P.bp_scratch + (size_t) blockIdx.x * 32 * G;
    float2 L[NP];
#pragma unroll
    for (int k = 0; k < NP; ++k) L[k] = P.lutq[tid + NT * k];
    const uint32_t kC = 0x9e3779b1u;
    const uint32_t x0 = (uint32_t) (SPT * tid) * kC;

    for (int64_t b = blockIdx.x; b < P.nb; b += gridDim.x) {
        if (!P.active[b]) continue;
        int fc = -1;
        if (P.cin) {
            fc = P.cin[b];
            if (fc < 0) continue;
        }
        if (tid < 128) {
            const int p = tid >> 2, j = tid & 3;
            const int wi = widx_d(p, j);
            const float dd = h2f(P.dcur[b]);
            Tt[p][j] = P.w[b * 128 + wi] / dd;
            Hh[p][j] = P.h ? P.h[b * 128 + wi] : 1.0f;
        }
        __syncthreads();

        float D[SPT];
        // tables of step q into parity q & 1
        auto build = [&](int q) {
            if constexpr (COMP) return;
            const int p = (P.rot + q) & 31;
            const float t0 = Tt[p][0], t1 = Tt[p][1], t2 = Tt[p][2], t3 = Tt[p][3];
            const float h0 = Hh[p][0], h1 = Hh[p][1], h2 = Hh[p][2], h3 = Hh[p][3];
#pragma unroll
            for (int k = 0; k < NP; ++k) {
                const float a0 = t0 - L[k].x, a1 = t1 - L[k].y, a2 = t2 - L[k].x, a3 = t3 - L[k].y;
                tab[q & 1][0][tid + NT * k] = h0 * a0 * a0 + h1 * a1 * a1;
                tab[q & 1][1][tid + NT * k] = h2 * a2 * a2 + h3 * a3 * a3;
            }
        };
        build(0);
        __syncthreads();
        {
            float tc[4], hc[4];
            if constexpr (COMP) {
                const int p0 = P.rot & 31;
                for (int i = 0; i < 4; ++i) { tc[i] = Tt[p0][i]; hc[i] = Hh[p0][i]; }
            }
#pragma unroll
            for (int j = 0; j < SPT; ++j) {
                const uint32_t x = x0 + (uint32_t) j * kC;
                float v;
                if constexpr (COMP) v = comp_cost((uint32_t) (SPT * tid + j), tc, hc);
                else v = tab[0][0][x >> 21] + tab[0][1][(x >> 10) & 2047u];
                if (fc >= 0 && (((SPT * tid + j) & (G - 1)) != fc)) v = kInf;
                D[j] = v;
            }
        }
        for (int q = 1; q < 32; ++q) {
            const int par = q & 1;
            build(q);
            // group minima of the previous column
            uint16_t* bpq = bp + (size_t) q * G;
            if constexpr (SPT >= SPAN) {
#pragma unroll
                for (int g = 0; g < GPT; ++g) {
                    float mv = D[g * SPAN];
                    int ma = 0;
#pragma unroll
                    for (int e = 1; e < SPAN; ++e) {
                        const float v = D[g * SPAN + e];
                        if (v < mv) { mv = v; ma = e; }
                    }
                    const int a = ((SPT * tid) >> K) + g;
                    mbuf[par][a] = mv;
                    bpq[a] = (uint16_t) (SPT * tid + g * SPAN + ma);
                }
            } else {
                // a group spans PC consecutive threads (the same wave): min/argmin of the thread, then across the PC
                float mv = D[0];
                int ma = 0;
#pragma unroll
                for (int e = 1; e < SPT; ++e) {
                    const float v = D[e];
                    if (v < mv) { mv = v; ma = e; }
                }
                int st = SPT * tid + ma;
#pragma unroll
                for (int off = 1; off < PC; off <<= 1) {
                    const float ov = __shfl_xor(mv, off);
                    const int os = __shfl_xor(st, off);
                    if (ov < mv || (ov == mv && os < st)) { mv = ov; st = os; }
                }
                if ((tid & (PC - 1)) == 0) {
                    const int a = (SPT * tid) >> K;
                    mbuf[par][a] = mv;
                    bpq[a] = (uint16_t) st;
                }
            }
            __syncthreads();
            const float4* m4 = reinterpret_cast<const float4*>(&mbuf[par][(SPT * tid) & (G - 1)]);
            // keep the compiler from caching the 2 * SPT hashed table addresses across the steps (they are loop invariant,
            // and 128 live registers spill): an empty asm makes x0 "new" every step
            uint32_t x0s = x0;
            asm volatile("" : "+v"(x0s));
            float tc[4], hc[4];
            if constexpr (COMP) {
                const int pq = (P.rot + q) & 31;
                for (int i = 0; i < 4; ++i) { tc[i] = Tt[pq][i]; hc[i] = Hh[pq][i]; }
            }
#pragma unroll
            for (int j4 = 0; j4 < SPT / 4; ++j4) {
                const float4 mm = m4[j4];
                const uint32_t xa = x0s + (uint32_t) (4 * j4) * kC;
                const uint32_t xb = xa + kC, xc = xb + kC, xd = xc + kC;
                if constexpr (COMP) {
                    const uint32_t sa = (uint32_t) (SPT * tid + 4 * j4);
                    D[4 * j4 + 0] = mm.x + comp_cost(sa, tc, hc);
                    D[4 * j4 + 1] = mm.y + comp_cost(sa + 1, tc, hc);
                    D[4 * j4 + 2] = mm.z + comp_cost(sa + 2, tc, hc);
                    D[4 * j4 + 3] = mm.w + comp_cost(sa + 3, tc, hc);
                } else {
                    D[4 * j4 + 0] = mm.x + tab[par][0][xa >> 21] + tab[par][1][(xa >> 10) & 2047u];
                    D[4 * j4 + 1] = mm.y + tab[par][0][xb >> 21] + tab[par][1][(xb >> 10) & 2047u];
                    D[4 * j4 + 2] = mm.z + tab[par][0][xc >> 21] + tab[par][1][(xc >> 10) & 2047u];
                    D[4 * j4 + 3] = mm.w + tab[par][0][xd >> 21] + tab[par][1][(xd >> 10) & 2047u];
                }
                if ((j4 & 3) == 3) __builtin_amdgcn_sched_barrier(0);   // 16 states at a time: do not batch all the gathers
            }
        }
        // final state
        const int lo = fc < 0 ? 0 : (fc << K), hi = fc < 0 ? 32768 : lo + SPAN;
        float bv = 3e38f;
        int bi = -1;
#pragma unroll
        for (int j = 0; j < SPT; ++j) {
            const int s = SPT * tid + j;
            if (s >= lo && s < hi && D[j] < bv) { bv = D[j]; bi = s; }
        }
#pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            const float ov = __shfl_xor(bv, off);
            const int oi = __shfl_xor(bi, off);
            if (ov < bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { redv[tid >> 5] = bv; redi[tid >> 5] = bi; }
        __syncthreads();
        if (tid == 0) {
            float v = redv[0];
            int i = redi[0];
            for (int w2 = 1; w2 < NT / 32; ++w2)
                if (redv[w2] < v || (redv[w2] == v && redi[w2] >= 0 && (i < 0 || redi[w2] < i))) { v = redv[w2]; i = redi[w2]; }
            // traceback
            if (i < 0 || v >= kInf * 0.5f) {
                if (P.cin) P.cost_out[b] = kInf; else P.c_out[b] = 0;
            } else if (!P.cin) {
                const int qj = (31 - P.rot) & 31;   // the step that processes position 31
                int s = i;
                for (int q = 31; q > qj; --q) s = bp[(size_t) q * G + (s & (G - 1))];
                P.c_out[b] = s >> K;
            } else {
                int s = i;
                P.st_out[b * 32 + ((P.rot + 31) & 31)] = (uint16_t) s;
                for (int q = 31; q >= 1; --q) {
                    s = bp[(size_t) q * G + (s & (G - 1))];
                    P.st_out[b * 32 + ((P.rot + q - 1) & 31)] = (uint16_t) s;
                }
                P.cost_out[b] = v;
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------ per-block stages

__device__ __forceinline__ void state_cb(const float2* lutq, int comp, uint32_t s, float* c) {
    if (comp) { comp_vals(s, c); return; }
    const uint32_t x = s * 0x9e3779b1u;
    const float2 a = lutq[x >> 21], b = lutq[(x >> 10) & 2047u];
    c[0] = a.x; c[1] = a.y; c[2] = b.x; c[3] = b.y;
}

__global__ void init_kernel(const float* w, const float* h, int64_t nb, uint16_t* dcur, uint8_t* active, uint8_t* status,
                            double* sumhw2, double mult, int first, uint16_t fixed_scale, double cb_rms) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    double s2 = 0, sh2 = 0, amax = 0;
    bool fin = true;
    for (int i = 0; i < 128; ++i) {
        const float x = w[b * 128 + i];
        const float hi = h ? h[b * 128 + i] : 1.0f;
        if (!isfinite(x) || !isfinite(hi)) fin = false;
        s2 += (double) x * x;
        sh2 += (double) hi * x * x;
        amax = fmax(amax, fabs((double) x));
    }
    if (first) {
        status[b] = !fin ? 2 : (amax == 0.0 ? 1 : 0);
        sumhw2[b] = sh2;
    }
    if (status[b] != 0) { active[b] = 0; return; }
    const double rms = sqrt(s2 / 128.0);
    double dd = rms / cb_rms * mult;
    uint16_t dh = f2h((float) fmin(dd, 65504.0));
    if (dh == 0) dh = 1;
    if (fixed_scale) dh = fixed_scale;
    dcur[b] = dh;
    active[b] = 1;
}

// error of (states by position st, fp16 scale dh)
__device__ double fit_d(const float* w, const float* h, int64_t b, const uint16_t* st, uint16_t dh, const float2* lutq, int comp) {
    const float d = h2f(dh);
    double e = 0;
    for (int p = 0; p < 32; ++p) {
        float c[4];
        state_cb(lutq, comp, st[p], c);
        for (int j = 0; j < 4; ++j) {
            const int wi = widx_d(p, j);
            const float rec = d * c[j];
            const double df = (double) w[b * 128 + wi] - rec;
            e += (h ? (double) h[b * 128 + wi] : 1.0) * df * df;
        }
    }
    return e;
}

// pick the best candidate, then the scale refinement step of encode_block
__global__ void refine_kernel(const float* w, const float* h, int64_t nb, int nc, const uint16_t* st_c, const float* cost_c,
                              uint16_t* dcur, uint8_t* active, uint16_t* best_d, double* best_err, uint16_t* best_st,
                              const float2* lutq, int fixed_scale, int comp) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb || !active[b]) return;
    int bc = -1;
    float bcost = kInf * 0.5f;
    for (int r = 0; r < nc; ++r) {
        const float c = cost_c[(int64_t) r * nb + b];
        if (c < bcost) { bcost = c; bc = r; }
    }
    if (bc < 0) { active[b] = 0; return; }
    uint16_t st[32];
    for (int p = 0; p < 32; ++p) st[p] = st_c[((int64_t) bc * nb + b) * 32 + p];
    const uint16_t dh = dcur[b];
    const double e = fit_d(w, h, b, st, dh, lutq, comp);
    if (e < best_err[b]) {
        best_err[b] = e;
        best_d[b] = dh;
        for (int p = 0; p < 32; ++p) best_st[b * 32 + p] = st[p];
    }
    if (e == 0.0 || fixed_scale) { active[b] = 0; return; }
    double num = 0, den = 0;
    for (int p = 0; p < 32; ++p) {
        float c[4];
        state_cb(lutq, comp, st[p], c);
        for (int j = 0; j < 4; ++j) {
            const int wi = widx_d(p, j);
            const double hi = h ? (double) h[b * 128 + wi] : 1.0, cc = c[j];
            num += hi * w[b * 128 + wi] * cc;
            den += hi * cc * cc;
        }
    }
    if (den <= 0 || num <= 0) { active[b] = 0; return; }
    const uint16_t nh = f2h((float) fmin(num / den, 65504.0));
    if (nh == 0 || nh == dh) { active[b] = 0; return; }
    const double en = fit_d(w, h, b, st, nh, lutq, comp);
    if (en < best_err[b]) {
        best_err[b] = en;
        best_d[b] = nh;
        for (int p = 0; p < 32; ++p) best_st[b * 32 + p] = st[p];
    }
    if (en >= e) { active[b] = 0; return; }
    dcur[b] = nh;
}

__global__ void dedupe_kernel(int64_t nb, int nc, int* c_all) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    for (int r = 1; r < nc; ++r) {
        const int c = c_all[(int64_t) r * nb + b];
        for (int q = 0; q < r; ++q)
            if (c_all[(int64_t) q * nb + b] == c) { c_all[(int64_t) r * nb + b] = -1; break; }
    }
}

__global__ void pack_kernel(int K, int64_t nb, const uint8_t* status, const double* sumhw2, const uint16_t* best_d,
                            const double* best_err, const uint16_t* best_st, uint8_t* dst, double* err_out) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const int bb = 2 + 4 * K;
    uint8_t* o = dst + b * bb;
    for (int i = 0; i < bb; ++i) o[i] = 0;
    if (status[b] == 2) { err_out[b] = NAN; return; }
    if (status[b] == 1 || best_err[b] >= sumhw2[b]) { err_out[b] = status[b] == 1 ? 0.0 : sumhw2[b]; return; }
    o[0] = (uint8_t) (best_d[b] & 255);
    o[1] = (uint8_t) (best_d[b] >> 8);
    uint8_t* qs = o + 2;
    for (int u = 0; u < 32; ++u) {
        const uint32_t x = best_st[b * 32 + u] & ((1u << K) - 1u);
        for (int bit = 0; bit < K; ++bit)
            if ((x >> bit) & 1u) qs[(u * K + bit) >> 3] |= (uint8_t) (1u << ((u * K + bit) & 7));
    }
    err_out[b] = best_err[b];
}

__global__ void init_best_kernel(int64_t nb, double* best_err, uint8_t* have) {
    const int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (b >= nb) return;
    best_err[b] = 1e300;
}

template <int K, bool COMP>
void launch_pass(const Pass& P, int grid) {
    hipLaunchKernelGGL((pass_kernel<K, kSL, COMP>), dim3(grid), dim3(NT), 0, 0, P);
}

void launch_pass_k(int K, bool comp, const Pass& P, int grid) {
    if (comp) {   // TQK5X / TQK4X
        if (K == 4) launch_pass<4, true>(P, grid);
        else launch_pass<5, true>(P, grid);
        return;
    }
    switch (K) {
        case 4: launch_pass<4, false>(P, grid); break;
        case 5: launch_pass<5, false>(P, grid); break;
        case 6: launch_pass<6, false>(P, grid); break;
        default: launch_pass<7, false>(P, grid); break;
    }
}

struct Dev {
    int device = 0;
    int nsm = 0;
    float2* lutq = nullptr;
    bool ready = false;
} g_dev;

void ensure_dev(int device) {
    if (g_dev.ready && g_dev.device == device) return;
    HCHECK(hipSetDevice(device));
    hipDeviceProp_t pr;
    HCHECK(hipGetDeviceProperties(&pr, device));
    g_dev.device = device;
    g_dev.nsm = pr.multiProcessorCount;
    // the 2048 codebook points as the decoder sees them: round(fp16 * 8192) / 8192 (tq_cpu.cpp fill_table)
    std::vector<float2> pts(2048);
    const float* pt = tqenc::point_table();
    for (int i = 0; i < 2048; ++i) pts[i] = make_float2(pt[2 * i], pt[2 * i + 1]);
    if (g_dev.lutq) HCHECK(hipFree(g_dev.lutq));
    HCHECK(hipMalloc(&g_dev.lutq, 2048 * sizeof(float2)));
    HCHECK(hipMemcpy(g_dev.lutq, pts.data(), 2048 * sizeof(float2), hipMemcpyHostToDevice));
    g_dev.ready = true;
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    HCHECK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

}  // namespace

extern "C" {
#if defined(_WIN32)
#define TQH_EXPORT __declspec(dllexport)
#else
#define TQH_EXPORT __attribute__((visibility("default")))
#endif

// GPU encode of nblocks blocks of TQK4X / 5X / 6 / 7.  Same arguments as tqe_encode (iters, ncand, scale_tries; exact
// search is CPU only).  stats[0] = nonfinite blocks, stats[1] = zero blocks; timing[0] = kernel seconds, [1] = passes.
TQH_EXPORT double tqh_encode(int type, const float* w, const float* h, int64_t nblocks, uint8_t* dst, int iters, int ncand,
                             int scale_tries, int device, double* err_out, int64_t* stats, double* timing,
                             int fixed_scale_bits) {
    const int K = tqenc::type_kbits(type);
    if (K < 4 || K > 7) return -1.0;
    const bool comp = tqenc::type_computed(type);
    const double cb_rms = tqenc::codebook_rms(type);
    ensure_dev(device);
    const int bb = 2 + 4 * K, G = 1 << (15 - K);
    const int nc = std::max(1, std::min(ncand, 8));
    static const int rots[8] = {16, 8, 24, 12, 20, 4, 28, 0};
    static const double mult[7] = {1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3};
    const int tries = fixed_scale_bits ? 1 : std::max(1, std::min(scale_tries, 7));
    const int64_t CH = 1 << 16;
    const int grid_cap = g_dev.nsm * 4;
    uint16_t* bp = dalloc<uint16_t>((size_t) grid_cap * 32 * G);
    double tot = 0;
    int64_t nf = 0, zb = 0;
    double ksec = 0;
    int64_t npass = 0;
    for (int64_t c0 = 0; c0 < nblocks; c0 += CH) {
        const int64_t nb = std::min(CH, nblocks - c0);
        float* dw = dalloc<float>((size_t) nb * 128);
        float* dh = h ? dalloc<float>((size_t) nb * 128) : nullptr;
        HCHECK(hipMemcpy(dw, w + c0 * 128, (size_t) nb * 128 * 4, hipMemcpyHostToDevice));
        if (h) HCHECK(hipMemcpy(dh, h + c0 * 128, (size_t) nb * 128 * 4, hipMemcpyHostToDevice));
        uint16_t* dcur = dalloc<uint16_t>(nb);
        uint8_t* active = dalloc<uint8_t>(nb);
        uint8_t* status = dalloc<uint8_t>(nb);
        double* sumhw2 = dalloc<double>(nb);
        uint16_t* best_d = dalloc<uint16_t>(nb);
        double* best_err = dalloc<double>(nb);
        uint16_t* best_st = dalloc<uint16_t>((size_t) nb * 32);
        uint16_t* st_c = dalloc<uint16_t>((size_t) nc * nb * 32);
        float* cost_c = dalloc<float>((size_t) nc * nb);
        int* c_all = dalloc<int>((size_t) nc * nb);
        uint8_t* d_dst = dalloc<uint8_t>((size_t) nb * bb);
        double* d_err = dalloc<double>(nb);
        uint8_t* have = nullptr;
        const int tb = 128, gb = (int) ((nb + tb - 1) / tb);
        hipLaunchKernelGGL(init_best_kernel, dim3(gb), dim3(tb), 0, 0, nb, best_err, have);
        const auto t0 = std::chrono::steady_clock::now();
        for (int tr = 0; tr < tries; ++tr) {
            hipLaunchKernelGGL(init_kernel, dim3(gb), dim3(tb), 0, 0, dw, dh, nb, dcur, active, status, sumhw2, mult[tr],
                               tr == 0 ? 1 : 0, (uint16_t) fixed_scale_bits, cb_rms);
            for (int it = 0; it <= (fixed_scale_bits ? 0 : iters); ++it) {
                Pass P{};
                P.w = dw; P.h = dh; P.dcur = dcur; P.active = active; P.nb = nb; P.lutq = g_dev.lutq; P.bp_scratch = bp;
                const int grid = (int) std::min<int64_t>(nb, grid_cap);
                for (int r = 0; r < nc; ++r) {
                    P.rot = rots[r]; P.cin = nullptr; P.c_out = c_all + (int64_t) r * nb;
                    launch_pass_k(K, comp, P, grid);
                }
                hipLaunchKernelGGL(dedupe_kernel, dim3(gb), dim3(tb), 0, 0, nb, nc, c_all);
                HCHECK(hipMemset(cost_c, 0x7f, (size_t) nc * nb * sizeof(float)));   // skipped candidates cost "infinity"
                for (int r = 0; r < nc; ++r) {
                    P.rot = 0; P.cin = c_all + (int64_t) r * nb; P.c_out = nullptr;
                    P.st_out = st_c + (int64_t) r * nb * 32; P.cost_out = cost_c + (int64_t) r * nb;
                    launch_pass_k(K, comp, P, grid);
                }
                hipLaunchKernelGGL(refine_kernel, dim3(gb), dim3(tb), 0, 0, dw, dh, nb, nc, st_c, cost_c, dcur, active,
                                   best_d, best_err, best_st, g_dev.lutq, fixed_scale_bits, comp ? 1 : 0);
                npass += 2 * nc;
                if (std::getenv("TQH_DEBUG")) {
                    HCHECK(hipDeviceSynchronize());
                    int hc[4] = {}; float hcost[4] = {}; double he[4] = {}; uint16_t hd[4] = {}; uint8_t ha[4] = {};
                    HCHECK(hipMemcpy(hc, c_all, 16, hipMemcpyDeviceToHost));
                    HCHECK(hipMemcpy(hcost, cost_c, 16, hipMemcpyDeviceToHost));
                    HCHECK(hipMemcpy(he, best_err, 32, hipMemcpyDeviceToHost));
                    HCHECK(hipMemcpy(hd, dcur, 8, hipMemcpyDeviceToHost));
                    HCHECK(hipMemcpy(ha, active, 4, hipMemcpyDeviceToHost));
                    for (int i = 0; i < 4; ++i)
                        std::fprintf(stderr, "[dbg try %d it %d] block %d: c %d cost %g best_err %g dcur %g active %d\n", tr, it, i, hc[i], hcost[i], he[i],
                                     tqenc::half_to_float(hd[i]), ha[i]);
                }
            }
        }
        hipLaunchKernelGGL(pack_kernel, dim3(gb), dim3(tb), 0, 0, K, nb, status, sumhw2, best_d, best_err, best_st, d_dst, d_err);
        HCHECK(hipDeviceSynchronize());
        ksec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        HCHECK(hipMemcpy(dst + (size_t) c0 * bb, d_dst, (size_t) nb * bb, hipMemcpyDeviceToHost));
        std::vector<double> herr(nb);
        std::vector<uint8_t> hst(nb);
        HCHECK(hipMemcpy(herr.data(), d_err, nb * 8, hipMemcpyDeviceToHost));
        HCHECK(hipMemcpy(hst.data(), status, nb, hipMemcpyDeviceToHost));
        for (int64_t i = 0; i < nb; ++i) {
            if (hst[i] == 2) ++nf;
            else tot += herr[i];
            if (hst[i] == 1) ++zb;
            if (err_out) err_out[c0 + i] = herr[i];
        }
        for (void* p : {(void*) dw, (void*) dh, (void*) dcur, (void*) active, (void*) status, (void*) sumhw2, (void*) best_d,
                        (void*) best_err, (void*) best_st, (void*) st_c, (void*) cost_c, (void*) c_all, (void*) d_dst,
                        (void*) d_err})
            if (p) HCHECK(hipFree(p));
    }
    HCHECK(hipFree(bp));
    if (stats) { stats[0] = nf; stats[1] = zb; }
    if (timing) { timing[0] = ksec; timing[1] = (double) npass; }
    return tot;
}
}
