// Included by kernels.cu right after the gfx11 chunked-GDN kernels (it uses their constants, types and helpers).
//
// STRATA_GDN_CHUNKED=1 on gfx12 (RDNA4: R9700 / 9070 / 9060): the same chunked algorithm as the gfx11 kernels above, on
// the gfx12 matrix cores.  The fragment layout differs from gfx11 (wave32, v_wmma_f32_16x16x16_f16; checked on an R9700 by
// results/gdn/GDN-A/micro/wmma12.cpp): A and B take 8 halves a lane, lane l holding row (A) or column (B) l & 15 and
// k = 8 * (l >> 4) + 0..7; the accumulator lane l holds column l & 15 and rows 8 * (l >> 4) + 0..7.  So an accumulator IS the
// next product's B operand (an FP16 conversion only: no permlane / byte permute as on gfx11), and the row of accumulator
// element v is 8 * hi + v (gfx11: 2 * v + hi).  Everything else (chunk of 16, FP16 operands, FP32 accumulate and state, the
// norm fused, T and P from a prep kernel) is the gfx11 design, so the numerics are the same class (FP16 operands).
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_GDN_WMMA12 1
#endif
__device__ __forceinline__ gw_f8 gw_mma12(gw_h8 a, gw_h8 b, gw_f8 c) {
#if defined(STRATA_GDN_WMMA12)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    (void) a; (void) b; (void) c;
    __builtin_trap();   // not a gfx12 device pass: the runtime gate never launches it
#endif
}
__device__ __forceinline__ gw_h8 gw_acc_to_b12(gw_f8 x) {
    gw_h8 r;
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = (_Float16) x[i];
    return r;
}
// A barrier over LDS only (LB): __syncthreads() is a workgroup fence over all memory, which also waits for the global loads of the
// next chunk that the scan keeps in flight (the same reason gfx11 has gw_lds_barrier).  gfx12: wait for this wave's LDS accesses
// (DScnt), then the barrier; the compiler still waits for each load before its first use.
template<bool LB> __device__ __forceinline__ void gw12_barrier() {
#if defined(STRATA_GDN_WMMA12)
    if (LB) {
        __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
        __builtin_amdgcn_s_barrier();
        __asm__ volatile("" ::: "memory");
    } else {
        __syncthreads();
    }
#else
    __syncthreads();
#endif
}
__device__ __forceinline__ gw_h8 gw_ld8(const _Float16* p) { return *reinterpret_cast<const gw_h8*>(p); }   // 16 bytes of LDS

// The causal conv + SiLU + q/k L2 norm (gdn_conv_l2_kernel's arithmetic) with the result written as FP16: half the traffic of
// the FP32 h in the conv and in both chunked kernels.  One block per (head of 128 channels, tile of 64 tokens).
__global__ void __launch_bounds__(128) gdn_conv_l2_h16_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                                              const float* __restrict__ w, _Float16* __restrict__ h, int64_t T,
                                                              float eps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool norm = blockIdx.x < 2 * HK;   // q and k heads; v channels are not normalised
    __shared__ float part[16][4];
    for (int64_t tb = t0; tb < t1; tb += 16) {
        const int n = (int) (t1 - tb < 16 ? t1 - tb : 16);
        float val[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            val[j] = 0.0f;
            if (j < n) {
                const float x = qkv[(tb + j) * C + c];
                const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
                val[j] = s / (1.0f + __expf(-s));
                v0 = v1; v1 = v2; v2 = x;
            }
        }
        if (norm) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                float sq = val[j] * val[j];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor(sq, o);
                if ((threadIdx.x & 31) == 0) part[j][threadIdx.x >> 5] = sq;
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < 16; ++j) val[j] *= rsqrtf(part[j][0] + part[j][1] + part[j][2] + part[j][3] + eps);
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (j < n) h[(tb + j) * C + c] = (_Float16) val[j];
    }
}

// H16: h is the FP16 conv output of gdn_conv_l2_h16_kernel (the matrix-core operands are FP16 anyway; v keeps 11 bits, the
// same rounding the y operand takes a step later), else the FP32 h of gdn_conv.
template<bool H16>
__global__ void __launch_bounds__(32 * GW_VPK) gdn_w12_prep_kernel(const float* __restrict__ hf, const _Float16* __restrict__ hh,
                                                                const float* __restrict__ gate,
                                                                const float* __restrict__ beta, _Float16* __restrict__ tp,
                                                                int64_t T) {
    __shared__ float sa[GW_VPK][GW_CK * GW_CK], sg[GW_VPK][GW_CK], sb[GW_VPK][GW_CK];
    const int qh = blockIdx.x, ch = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int vh = qh + w * HK;
    const int64_t t0 = (int64_t) ch * GW_CK;
    gw_f8 kk = {}, qk = {};   // k_t.k_i and q_t.k_i (lane: column i = c, rows t = 8 * hi + v)
    {
        const bool ok = t0 + c < T;   // past the end: zero rows
        const float* ht = hf + (ok ? t0 + c : 0) * C;
        const _Float16* hth = hh + (ok ? t0 + c : 0) * C;
#pragma unroll
        for (int j = 0; j < S / 16; ++j) {
            gw_h8 ka, qa;
            if (H16) {
                ka = *reinterpret_cast<const gw_h8*>(hth + HK * S + qh * S + 16 * j + 8 * hi);
                qa = *reinterpret_cast<const gw_h8*>(hth + qh * S + 16 * j + 8 * hi);
                if (!ok) { ka = gw_h8{}; qa = gw_h8{}; }
            } else {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const gw_f4 k4 = *reinterpret_cast<const gw_f4*>(ht + HK * S + qh * S + 16 * j + 8 * hi + 4 * i);
                    const gw_f4 q4 = *reinterpret_cast<const gw_f4*>(ht + qh * S + 16 * j + 8 * hi + 4 * i);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        ka[4 * i + e] = (_Float16) (ok ? k4[e] : 0.0f);
                        qa[4 * i + e] = (_Float16) (ok ? q4[e] : 0.0f);
                    }
                }
            }
            kk = gw_mma12(ka, ka, kk);   // B = K^T: lane i holds column i of K^T = row i of K, the A operand itself
            qk = gw_mma12(qa, ka, qk);
        }
    }
    if (l < GW_CK) {   // past the end: gate 0, beta 0
        const bool ok = t0 + l < T;
        float g = ok ? gate[(t0 + l) * HV + vh] : 0.0f;
#pragma unroll
        for (int o = 1; o < GW_CK; o <<= 1) {
            const float u = __shfl_up(g, o, GW_CK);
            if (l >= o) g += u;
        }
        sg[w][l] = g;
        sb[w][l] = ok ? beta[(t0 + l) * HV + vh] : 0.0f;
    }
    __syncthreads();
    _Float16* out = tp + ((int64_t) ch * HV + vh) * (2 * GW_CK * GW_CK);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int t = 8 * hi + v;
        sa[w][t * GW_CK + c] = c < t ? sb[w][t] * __expf(sg[w][t] - sg[w][c]) * kk[v] : 0.0f;
        out[GW_CK * GW_CK + t * GW_CK + c] = (_Float16) (c <= t ? __expf(sg[w][t] - sg[w][c]) * qk[v] : 0.0f);
    }
    __syncthreads();
    if (l < GW_CK) {   // lane c: column c of T = (I + A)^-1 by forward substitution
        float tc[GW_CK];
#pragma unroll
        for (int t = 0; t < GW_CK; ++t) {
            float v = t == c ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < t; ++k) v = fmaf(-sa[w][t * GW_CK + k], tc[k], v);
            tc[t] = v;
            out[t * GW_CK + c] = (_Float16) v;
        }
    }
}

// gw_load / gw_touch for the FP16 h (a thread's 8 channels of k, q, v are one 16-byte load each)
__device__ __forceinline__ void gw12_load16(GwChunk& in, const _Float16* __restrict__ hh, const float* __restrict__ gate,
                                            const float* __restrict__ beta, const unsigned* __restrict__ tp, int64_t t0, int64_t T,
                                            int vh, int qh) {
    const int t = threadIdx.x >> 4, r = (threadIdx.x & 15) * 8;
    const _Float16* ht = hh + std::min<int64_t>(t0 + t, T - 1) * C;
    in.tp = tp[((t0 / GW_CK) * HV + vh) * (GW_CK * GW_CK) + threadIdx.x];
    const gw_h8 k = *reinterpret_cast<const gw_h8*>(ht + HK * S + qh * S + r);
    const gw_h8 q = *reinterpret_cast<const gw_h8*>(ht + qh * S + r);
    const gw_h8 v = *reinterpret_cast<const gw_h8*>(ht + 2 * HK * S + vh * S + r);
    in.k0 = gw_f4{(float) k[0], (float) k[1], (float) k[2], (float) k[3]};
    in.k1 = gw_f4{(float) k[4], (float) k[5], (float) k[6], (float) k[7]};
    in.q0 = gw_f4{(float) q[0], (float) q[1], (float) q[2], (float) q[3]};
    in.q1 = gw_f4{(float) q[4], (float) q[5], (float) q[6], (float) q[7]};
    in.v0 = gw_f4{(float) v[0], (float) v[1], (float) v[2], (float) v[3]};
    in.v1 = gw_f4{(float) v[4], (float) v[5], (float) v[6], (float) v[7]};
    if (threadIdx.x < GW_CK) {
        const int64_t tl = std::min<int64_t>(t0 + threadIdx.x, T - 1);
        in.g = gate[tl * HV + vh];
        in.b = beta[tl * HV + vh];
    }
}
__device__ __forceinline__ unsigned gw12_touch16(const _Float16* __restrict__ hh, const float* __restrict__ z, int64_t t0, int64_t T,
                                                 int vh, int qh) {
    const int t = threadIdx.x >> 4, part = threadIdx.x & 15, kind = part >> 2;
    if (t0 + t >= T) return 0u;
    if (kind == 3) return __float_as_uint((z + (t0 + t) * HV * S + vh * S)[32 * (part & 3)]);
    const _Float16* p = kind == 0 ? hh + (t0 + t) * C + HK * S + qh * S
                      : kind == 1 ? hh + (t0 + t) * C + qh * S
                                  : hh + (t0 + t) * C + 2 * HK * S + vh * S;
    return *reinterpret_cast<const unsigned*>(p + 64 * (part & 1));   // 256 bytes of FP16: two lines
}

template<bool LB, bool H16>
__global__ void __launch_bounds__(GW_NW * 32) gdn_w12_scan_kernel(float* __restrict__ state, const float* __restrict__ hf,
                                                                  const _Float16* __restrict__ hh,
                                                                  const float* __restrict__ gate, const float* __restrict__ beta,
                                                                  const float* __restrict__ z, const float* __restrict__ gamma,
                                                                  const unsigned* __restrict__ tp, float eps,
                                                                  uint16_t* __restrict__ y16, int64_t T, int64_t ld16) {
    __shared__ __attribute__((aligned(16))) _Float16 sk[GW_CK * GW_KP], sq[GW_CK * GW_KP], skt[S * GW_CK],
        stp[2 * GW_CK * GW_CK];
    __shared__ __attribute__((aligned(16))) float sv[GW_CK * S], so[GW_CK * S], sn[GW_CK * GW_NW];
    __shared__ float sg[GW_CK], sb[GW_CK], sgam[S];
    const _Float16 *st_ = stp, *sp_ = stp + GW_CK * GW_CK;   // T, P
    const int vh = blockIdx.x, qh = vh % HK, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int col = 16 * w + c;
    const int st = threadIdx.x >> 4, sr = (threadIdx.x & 15) * 8;
    const size_t rs = (size_t) HV * S;
    gw_f8 s[S / 16];   // the state: rows 16j + 8 * hi + v, column col
#pragma unroll
    for (int j = 0; j < S / 16; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) s[j][v] = state[(size_t) (16 * j + 8 * hi + v) * rs + (size_t) vh * S + col];
    if (threadIdx.x < S) sgam[threadIdx.x] = gamma[threadIdx.x];
    const float out_scale = rsqrtf((float) S);
    GwChunk cur;
    if (H16) gw12_load16(cur, hh, gate, beta, tp, 0, T, vh, qh); else gw_load(cur, hf, gate, beta, tp, 0, T, vh, qh);
    unsigned touched = 0;
    gw_f4 z0 = {}, z1 = {};
    auto store_rows = [&](int64_t c0) {
        if (c0 + st >= T) return;
        const float* o = &so[st * S + sr];
        gw_h8 r;
#pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = (_Float16) (o[i] * sigm(i < 4 ? z0[i] : z1[i - 4]));
        *reinterpret_cast<uint4*>(y16 + (c0 + st) * ld16 + vh * S + sr) = __builtin_bit_cast(uint4, r);
    };
    for (int64_t t0 = 0; t0 < T; t0 += GW_CK) {
        {   // stage this chunk
            gw_h8 kh, qq;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                kh[i] = (_Float16) (i < 4 ? cur.k0[i] : cur.k1[i - 4]);
                qq[i] = (_Float16) (i < 4 ? cur.q0[i] : cur.q1[i - 4]);
                skt[(sr + i) * GW_CK + st] = kh[i];
            }
            *reinterpret_cast<uint4*>(&sk[st * GW_KP + sr]) = __builtin_bit_cast(uint4, kh);
            *reinterpret_cast<uint4*>(&sq[st * GW_KP + sr]) = __builtin_bit_cast(uint4, qq);
            *reinterpret_cast<gw_f4*>(&sv[st * S + sr]) = cur.v0;
            *reinterpret_cast<gw_f4*>(&sv[st * S + sr + 4]) = cur.v1;
            reinterpret_cast<unsigned*>(stp)[threadIdx.x] = cur.tp;
            if (threadIdx.x < GW_CK) {   // past the end: gate 0, beta 0
                const bool ok = t0 + l < T;
                float g = ok ? cur.g : 0.0f;
#pragma unroll
                for (int o = 1; o < GW_CK; o <<= 1) {
                    const float u = __shfl_up(g, o, GW_CK);
                    if (l >= o) g += u;
                }
                sg[l] = g;
                sb[l] = ok ? cur.b : 0.0f;
            }
        }
        if (t0 + GW_CK < T) {
            if (H16) gw12_load16(cur, hh, gate, beta, tp, t0 + GW_CK, T, vh, qh);
            else gw_load(cur, hf, gate, beta, tp, t0 + GW_CK, T, vh, qh);
        }
        const unsigned touch_now = H16 ? gw12_touch16(hh, z, t0 + GW_AHEAD * GW_CK, T, vh, qh)
                                       : gw_touch(hf, z, t0 + GW_AHEAD * GW_CK, T, vh, qh);
        gw12_barrier<LB>();   // (1) this chunk staged; the previous chunk's norm in so
        if (t0 > 0) store_rows(t0 - GW_CK);
        __asm__ volatile("" ::"v"(touched));
        touched = touch_now;
        {
            const int64_t tz = std::min<int64_t>(t0 + st, T - 1);
            z0 = *reinterpret_cast<const gw_f4*>(z + tz * HV * S + vh * S + sr);
            z1 = *reinterpret_cast<const gw_f4*>(z + tz * HV * S + vh * S + sr + 4);
        }
        gw_f8 x = {}, qs = {};   // K S0 and Q S0, this wave's columns
#pragma unroll
        for (int j = 0; j < S / 16; ++j) {
            const gw_h8 b = gw_acc_to_b12(s[j]);
            x = gw_mma12(gw_ld8(sk + c * GW_KP + 16 * j + 8 * hi), b, x);
            qs = gw_mma12(gw_ld8(sq + c * GW_KP + 16 * j + 8 * hi), b, qs);
        }
        gw_f8 y;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = 8 * hi + v;
            y[v] = sb[t] * (sv[t * S + col] - __expf(sg[t]) * x[v]);
        }
        const gw_f8 d = gw_mma12(gw_ld8(st_ + c * GW_CK + 8 * hi), gw_acc_to_b12(y), gw_f8{});
        gw_f8 o;
#pragma unroll
        for (int v = 0; v < 8; ++v) o[v] = __expf(sg[8 * hi + v]) * qs[v];
        o = gw_mma12(gw_ld8(sp_ + c * GW_CK + 8 * hi), gw_acc_to_b12(d), o);
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            o[v] *= out_scale;
            float p = o[v] * o[v];
#pragma unroll
            for (int m = 1; m < 16; m <<= 1) p += __shfl_xor(p, m);
            if (c == 0) sn[(8 * hi + v) * GW_NW + w] = p;
        }
        const float gl = sg[GW_CK - 1];
        gw_f8 dd;
#pragma unroll
        for (int v = 0; v < 8; ++v) dd[v] = __expf(gl - sg[8 * hi + v]) * d[v];
        const gw_h8 db = gw_acc_to_b12(dd);
        const float eg = __expf(gl);
#pragma unroll
        for (int j = 0; j < S / 16; ++j) s[j] = gw_mma12(gw_ld8(skt + (16 * j + c) * GW_CK + 8 * hi), db, s[j] * eg);
        gw12_barrier<LB>();   // (2) the sums of o^2
        const float g_col = sgam[col];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = 8 * hi + v;
            const gw_f4 n0 = *reinterpret_cast<const gw_f4*>(&sn[t * GW_NW]), n1 = *reinterpret_cast<const gw_f4*>(&sn[t * GW_NW + 4]);
            const float ss = n0[0] + n0[1] + n0[2] + n0[3] + n1[0] + n1[1] + n1[2] + n1[3];
            so[t * S + col] = o[v] * rsqrtf(ss * (1.0f / S) + eps) * g_col;
        }
    }
    __asm__ volatile("" ::"v"(touched));
    gw12_barrier<LB>();
    store_rows((T - 1) / GW_CK * GW_CK);
    __asm__ volatile("" : "+s"(state));
#pragma unroll
    for (int j = 0; j < S / 16; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) state[(size_t) (16 * j + 8 * hi + v) * rs + (size_t) vh * S + col] = s[j][v];
}
