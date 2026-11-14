// metal/kernels/gridbar.metal - feasibility probe for a persistent kernel: phases separated by an in-kernel grid
// barrier (a device atomic counter) instead of dispatch boundaries.  Metal does not promise that all threadgroups of
// a grid are resident at once; the probe launches at most one threadgroup per GPU core.
//
// Each phase p: every thread streams its share of chunk p (n uint4), and threadgroup g writes a token value
// (p * 1000 + g) to flag[p][g]; after the barrier, phase p + 1 checks every flag of phase p (written by the OTHER
// threadgroups) and counts mismatches into bad[0].

static inline void grid_barrier(device atomic_uint* bar, uint target, ushort tid) {
    threadgroup_barrier(mem_flags::mem_device);
#ifdef GB_FENCE
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
    if (tid == 0) {
        atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
        while (atomic_load_explicit(bar, memory_order_relaxed) < target) {}
    }
    threadgroup_barrier(mem_flags::mem_device);
#ifdef GB_FENCE
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
}

kernel void k_phases(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                     device atomic_uint* bar [[buffer(2)]], device volatile uint* flag [[buffer(3)]],
                     device atomic_uint* bad [[buffer(4)]], constant uint& n [[buffer(5)]],
                     constant uint& phases [[buffer(6)]],
                     uint tgpx [[threadgroup_position_in_grid]], uint ntgx [[threadgroups_per_grid]],
                     ushort tid [[thread_index_in_threadgroup]], uint gid [[thread_position_in_grid]],
                     uint nt [[threads_per_grid]]) {
    const uint g = tgpx, ntg = ntgx;
    uint4 acc = 0;
    for (uint p = 0; p < phases; ++p) {
        if (p > 0 && tid < ntg) {
            const uint v = flag[(p - 1) * ntg + tid];
            if (v != (p - 1) * 1000 + tid) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
        }
        device const uint4* s = src + ulong(p) * n;
        for (uint i = gid; i < n; i += nt) acc += s[i];
        if (tid == 0) flag[p * ntg + g] = p * 1000 + g;
        grid_barrier(bar, (p + 1) * ntg, tid);
    }
    out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

// one phase per dispatch, for comparison
kernel void k_phase1(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                     constant uint& n [[buffer(2)]], constant uint& p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) {
    uint4 acc = 0;
    device const uint4* s = src + ulong(p) * n;
    for (uint i = gid; i < n; i += nt) acc += s[i];
    out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
