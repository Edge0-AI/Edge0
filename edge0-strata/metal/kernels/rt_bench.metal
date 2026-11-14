// metal/kernels/rt_bench.metal - bandwidth and dispatch-cost probes (rt_bench.cpp).

// Grid-stride streaming read: thread i reads src[i], src[i + nt], ...  (coalesced across the grid).
kernel void k_read(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                   constant uint& n [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) {
    uint4 a = 0;
    for (uint i = gid; i < n; i += nt) a += src[i];
    out[gid] = a.x ^ a.y ^ a.z ^ a.w;
}

// The same, 4 independent loads in flight per iteration.
kernel void k_read_u4(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) {
    uint4 a = 0, b = 0, c = 0, d = 0;
    uint i = gid;
    for (; i + 3 * nt < n; i += 4 * nt) {
        a += src[i];
        b += src[i + nt];
        c += src[i + 2 * nt];
        d += src[i + 3 * nt];
    }
    for (; i < n; i += nt) a += src[i];
    a += b + c + d;
    out[gid] = a.x ^ a.y ^ a.z ^ a.w;
}

// Contiguous chunk per threadgroup (how a GEMV row block reads): threadgroup g reads [g*chunk, (g+1)*chunk).
kernel void k_read_chunk(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]], constant uint& chunk [[buffer(3)]],
                         uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]],
                         uint g [[threadgroup_position_in_grid]], uint gid [[thread_position_in_grid]]) {
    uint4 a = 0, b = 0;
    const uint base = g * chunk, end = min(base + chunk, n);
    uint i = base + tid;
    for (; i + tgs < end; i += 2 * tgs) {
        a += src[i];
        b += src[i + tgs];
    }
    for (; i < end; i += tgs) a += src[i];
    a += b;
    out[gid] = a.x ^ a.y ^ a.z ^ a.w;
}

// Read + write (a state update): dst[i] = src[i] * 1.0001 for float4.
kernel void k_copy(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant uint& n [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint nt [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += nt) dst[i] = src[i] * 1.0001f;
}

// Dispatch-cost probes.
kernel void k_empty(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0xFFFFFFFFu) out[0] = 1;
}

// A dependent chain: each dispatch reads what the previous one wrote.
kernel void k_chain(device uint* buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) buf[0] = buf[0] + 1;
}

// A small but real kernel: 2560 floats read, 2560 written (an elementwise op on one token).
kernel void k_small(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < 2560) y[gid] = x[gid] * 0.5f + y[gid];
}
