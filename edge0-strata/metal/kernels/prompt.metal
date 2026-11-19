// metal/kernels/prompt.metal - the elementwise steps of a prompt chunk around the GEMMs (after quant/units.metal):
//
//   route_topk     per token: softmax over the 512 logits, top 10 (ties: the lower expert), weights renormalised
//                  over the selected with a 2^-14 floor - moe_route's rule; one simdgroup a token
//   gr_epi_down    after the HC down GEMM (V = xn . [w_down; w_inject], T x (hc_lr + hc)): lo_h = half(silu(v / hc)),
//                  inject = v
//   gr_epi_up      after the HC up GEMM (G = lo . w_up, T x hc*n): mixed = mean_c xn[c] * sigmoid(G[c]) -> float and
//                  half (the next GEMMs' operand)
//   swiglu_h       h = half(silu(g) * u)
//   moe_out        bo = y_routed + sigmoid(sg) * y_shared

kernel void route_topk(device const float* logits [[buffer(0)]], device int* ids [[buffer(1)]],
                       device float* w [[buffer(2)]], constant uint& n_tok [[buffer(3)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x * 8 + sg;
    if (t >= n_tok) return;
    device const float* l = logits + ulong(t) * 512;
    float v[16];
    for (int i = 0; i < 16; ++i) v[i] = l[lane + 32 * i];
    float mx = -INFINITY;
    for (int i = 0; i < 16; ++i) mx = max(mx, v[i]);
    for (ushort m = 16; m > 0; m >>= 1) mx = max(mx, simd_shuffle_xor(mx, m));
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        v[i] = exp(v[i] - mx);
        sum += v[i];
    }
    for (ushort m = 16; m > 0; m >>= 1) sum += simd_shuffle_xor(sum, m);
    const float rcp = 1.0f / sum;
    for (int i = 0; i < 16; ++i) {
        v[i] *= rcp;
        if (isnan(v[i])) v[i] = -FLT_MAX;
    }
    float selected = 0.0f, ssum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
        for (int i = 1; i < 16; ++i)
            if (v[i] > best) { best = v[i]; ex = lane + 32 * i; }
        for (ushort m = 16; m > 0; m >>= 1) {
            const float ob = simd_shuffle_xor(best, m);
            const int oe = simd_shuffle_xor(ex, m);
            if (ob > best || (ob == best && oe < ex)) { best = ob; ex = oe; }
        }
        if ((ex & 31) == lane) {
            v[ex / 32] = -INFINITY;
            ids[t * 10 + rank] = ex;
            ssum += best;
        }
        if (rank == lane) selected = best;
    }
    for (ushort m = 16; m > 0; m >>= 1) ssum += simd_shuffle_xor(ssum, m);
    ssum = max(ssum, 6.103515625e-5f);
    if (lane < 10) w[t * 10 + lane] = selected * (1.0f / ssum);
}

struct EpiArgs {
    uint n_tok;
    uint n_embd;   // 2560
    uint hc;       // 4
    uint hc_lr;    // 320
};

// grid (T * (hc_lr + hc) / 256)
kernel void gr_epi_down(device const float* V [[buffer(0)]], device half* lo_h [[buffer(1)]],
                        device float* inject [[buffer(2)]], constant EpiArgs& a [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint nr = a.hc_lr + a.hc, t = gid / nr, r = gid % nr;
    if (t >= a.n_tok) return;
    const float v = V[gid];
    if (r < a.hc_lr) {
        const float z = v / float(a.hc);
        lo_h[t * a.hc_lr + r] = half(z / (1.0f + exp(-z)));
    } else {
        inject[t * a.hc + r - a.hc_lr] = v;
    }
}

// grid (T * n_embd / 256)
kernel void gr_epi_up(device const float* G [[buffer(0)]], device const float* xn [[buffer(1)]],
                      device float* mixed [[buffer(2)]], device half* mixed_h [[buffer(3)]],
                      constant EpiArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint n = a.n_embd, t = gid / n, d = gid % n;
    if (t >= a.n_tok) return;
    float m = 0.0f;
    for (uint c = 0; c < a.hc; ++c) {
        const float g = G[ulong(t) * a.hc * n + c * n + d];
        m = fma(xn[ulong(t) * a.hc * n + c * n + d], 1.0f / (1.0f + exp(-g)), m);
    }
    m = m / float(a.hc);
    mixed[gid] = m;
    mixed_h[gid] = half(m);
}

kernel void swiglu_h(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device half* h [[buffer(2)]],
                     constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const float gv = g[gid];
    h[gid] = half(gv / (1.0f + exp(-gv)) * u[gid]);
}

// grid (T * n_embd / 256); sg: T values
kernel void moe_out(device const float* yr [[buffer(0)]], device const float* ys [[buffer(1)]],
                    device const float* sg [[buffer(2)]], device float* bo [[buffer(3)]], constant uint2& a [[buffer(4)]],
                    uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.y;
    if (t >= a.x) return;
    bo[gid] = fma(1.0f / (1.0f + exp(-sg[t])), ys[gid], yr[gid]);
}

// ---- the HC read with fewer passes (after dense_mm.metal: GemmArgs, MetalPerformancePrimitives) ---------------------
// gr_norm_h: gr_norm with xn written as fp16 only (the GEMM operand; gemm_hc_up's epilogue reads it too).
// one threadgroup per (stream c, token t), 256 threads; args as gr_norm's (GrArgs, gr.metal)
kernel void gr_norm_h(device float* R [[buffer(0)]], device const float* bo [[buffer(1)]],
                      device const float* inj [[buffer(2)]], device const float* w_norm [[buffer(3)]],
                      device half* xnh [[buffer(4)]], constant GrArgs& a [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    device float* r = R + (t * a.hc + c) * n;
    const float g = a.apply ? 2.0f / (1.0f + exp(-inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    float ss = 0.0f;
    for (uint d = tid; d < n; d += 256) {
        float v = r[d];
        if (a.apply) {
            v = fma(bo[t * n + d], g, v);
            r[d] = v;
        }
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = rsqrt(tot / float(n) + a.eps);
    device half* o = xnh + (t * a.hc + c) * n;
    device const float* w = w_norm + c * n;
    for (uint d = tid; d < n; d += 256) o[d] = half(r[d] * w[d] * rs);
}

// gemm_hc_up: G = lo . w_up^T with w_up's rows interleaved by dq_f16 (perm = hc: row d * hc + c), so a TN-column
// tile holds TN / hc outputs d with all their streams; the epilogue mixed[t][d] = mean_c xn[t][c][d] *
// sigmoid(G[t][d * hc + c]) -> fp32 and fp16.  G is never written.  grid (hc * n / TN, ceil(T / TM)), 32 * SG threads.
template <int TM, int TN, int SG>
kernel void gemm_hc_up(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device const half* xnh [[buffer(2)]],
                       device float* mixed [[buffer(3)]], device half* mixh [[buffer(4)]], constant GemmArgs& g [[buffer(5)]],
                       constant EpiArgs& e [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float ot[TM * TN];
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(g.K, g.M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(g.K, g.N));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    op.run(mA, mB, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int hc = int(e.hc), n = int(e.n_embd), DPT = TN / hc;
    const int t0 = int(tg.y) * TM, d0 = int(tg.x) * DPT;
    for (int idx = tid; idx < TM * DPT; idx += SG * 32) {
        const int m = idx / DPT, dd = idx - m * DPT, t = t0 + m;
        if (t >= int(e.n_tok)) continue;
        float s = 0.0f;
        for (int c = 0; c < hc; ++c)
            s = fma(float(xnh[ulong(t) * hc * n + c * n + d0 + dd]), 1.0f / (1.0f + exp(-ot[m * TN + dd * hc + c])), s);
        s = s / float(hc);
        mixed[ulong(t) * n + d0 + dd] = s;
        mixh[ulong(t) * n + d0 + dd] = half(s);
    }
}
template [[host_name("gemm_hc_up_64_64_4")]] kernel decltype(gemm_hc_up<64, 64, 4>) gemm_hc_up<64, 64, 4>;
template [[host_name("gemm_hc_up_64_32_4")]] kernel decltype(gemm_hc_up<64, 32, 4>) gemm_hc_up<64, 32, 4>;
template [[host_name("gemm_hc_up_32_64_4")]] kernel decltype(gemm_hc_up<32, 64, 4>) gemm_hc_up<32, 64, 4>;
