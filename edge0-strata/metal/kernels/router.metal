// metal/kernels/router.metal - the router GEMV and moe_route in ONE dispatch (after quant/units/moe/gemv.metal): the
// logits rows as gemv's (LPR = 32 lanes a row, KS = 1: bitwise gemv_<F>_32_1_*_1's), 16 rows a threadgroup of 512
// threads; every threadgroup bumps a device counter after its rows (device-scope fences around it), and the LAST one
// runs moe_route's body (the softmax / top-10 / weights, then the plan) with its 512 threads (= n_expert) and resets
// the counter.  No threadgroup waits for another.  ids / w / plan are bitwise moe_route's.
// Buffers: W, xp, s16, logits, args (GemvArgs: n_out = n_expert), ids, w, plan, MoeArgs, counter (one uint, zero).

static inline void route_body(device const float* logits, device int* ids, device float* w, device int* plan,
                              constant MoeArgs& a, ushort tid, ushort nth, ushort sg, ushort lane, threadgroup int* s_grp,
                              threadgroup int* s_ent) {
    if (sg < a.n_tok) {
        device const float* l = logits + sg * 512;
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
                ids[sg * 10 + rank] = ex;
                ssum += best;
            }
            if (rank == lane) selected = best;
        }
        for (ushort m = 16; m > 0; m >>= 1) ssum += simd_shuffle_xor(ssum, m);
        ssum = max(ssum, 6.103515625e-5f);
        const float inv = 1.0f / ssum;
        if (lane < 10) w[sg * 10 + lane] = selected * inv;
    }
    threadgroup_barrier(mem_flags::mem_device);
    plan_body(ids, w, plan, a, tid, nth, sg, lane, s_grp, s_ent);
}

template <int F>
kernel void router_route(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                         device const float* s16 [[buffer(2)]], device float* logits [[buffer(3)]],
                         constant GemvArgs& ga [[buffer(4)]], device int* ids [[buffer(5)]], device float* w [[buffer(6)]],
                         device int* plan [[buffer(7)]], constant MoeArgs& a [[buffer(8)]],
                         device atomic_uint* counter [[buffer(9)]],
                         uint tgx [[threadgroup_position_in_grid]], uint ngx [[threadgroups_per_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort nth [[threads_per_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT;
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup int s_grp[32], s_ent[32];
    threadgroup uint last;
    load_tab<F>(tab, tid, nth);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the GEMV: simdgroup sg -> row tg.x * 16 + sg, 32 lanes over its units
    const int row = int(tgx) * 16 + sg;
    const int rr = min(row, int(ga.n_out) - 1);
    device const uchar* rp = W + (ulong)rr * ga.row_bytes;
    const int U = int(ga.n_in) / UNIT, s16n = int(ga.n_in) / 16;
    float acc[8];
    for (int t = 0; t < NTOK; ++t) acc[t] = 0.0f;
    for (int u = lane; u < U; u += 32) {
        float q[UNIT];
        float4 sc;
        Dec<F>::run(rp, u, q, sc, tab);
        for (int t = 0; t < NTOK; ++t) {
            float xs[UNIT], xsum[2];
            X<F>::load(xp + t * int(ga.x_stride), s16 + t * s16n, u, xs, xsum);
            acc[t] += unit_combine<UNIT>(q, sc, xs, xsum);
        }
    }
    for (int t = 0; t < NTOK; ++t) {
        const float v = lane_sum<32>(acc[t]);
        if (lane == 0 && row < int(ga.n_out)) logits[t * int(ga.y_stride) + row] = v;
    }
    // the last threadgroup to finish routes
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) last = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed) == ngx - 1 ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    route_body(logits, ids, w, plan, a, tid, nth, sg, lane, s_grp, s_ent);
    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

template [[host_name("router_route_30")]] kernel decltype(router_route<T_BF16>) router_route<T_BF16>;
template [[host_name("router_route_0")]] kernel decltype(router_route<T_F32>) router_route<T_F32>;
