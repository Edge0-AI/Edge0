// metal/tests/dsu_job.metal - test-only helpers appended to ds4_iq.metal / moe_mm.metal in dsu_parity.cpp.
// dsu_mkjob writes a Ds4MvJob whose pointers are inside the one buffer it was given (so everything the real kernel
// dereferences is a bound, resident buffer): mem + offsets.
#ifdef DSU_MKJOB
kernel void dsu_mkjob(device uchar* mem [[buffer(0)]], constant ulong* o [[buffer(1)]], constant int* a [[buffer(2)]]) {
    // o: job, w, xq, xd, y offsets; a: type, k, n, ncol, ld
    device Ds4MvJob& j = *(device Ds4MvJob*) (mem + o[0]);
    j.w = mem + o[1];
    j.ld = ulong(a[4]);
    j.type = a[0];
    j.k = a[1];
    j.n = a[2];
    j.ncol = a[3];
    j.xq = (device const char*) (mem + o[2]);
    j.xd = (device const float*) (mem + o[3]);
    j.y = (device float*) (mem + o[4]);
    j.ycs = ulong(a[2]);
    j.bias = nullptr;
    j.alpha = 1.0f;
}
#endif
#ifdef DSU_W8
// every 4-value run the tile loader reads (k and k + 16 of each 32-group's first half) for T, into half rows
template <int F>
static inline void dsu_w8_row(device const uchar* row, int k_total, device half* out, uint gid, threadgroup uint* tab) {
    const int runs = k_total / 32 * 4;   // per row: 4 runs of 4 values (k = 0,4,8,12 of the group), each giving k and k+16
    if (int(gid) >= runs) return;
    const int grp = int(gid) >> 2, r = int(gid) & 3;
    const int k = grp * 32 + r * 4;
    half4 a, b;
    W8<F>::run(row, k, tab, a, b);
    out[k] = a.x; out[k + 1] = a.y; out[k + 2] = a.z; out[k + 3] = a.w;
    out[k + 16] = b.x; out[k + 17] = b.y; out[k + 18] = b.z; out[k + 19] = b.w;
}
kernel void dsu_w8_iq1s(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant uint& k [[buffer(2)]],
                        constant uint& rb [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    threadgroup uint tab[1];
    dsu_w8_row<T_IQ1_S>(W + ulong(gid.y) * rb, int(k), out + ulong(gid.y) * k, gid.x, tab);
}
kernel void dsu_w8_iq1m(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant uint& k [[buffer(2)]],
                        constant uint& rb [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    threadgroup uint tab[1];
    dsu_w8_row<T_IQ1_M>(W + ulong(gid.y) * rb, int(k), out + ulong(gid.y) * k, gid.x, tab);
}
#endif
