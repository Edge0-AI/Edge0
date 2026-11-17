// metal/kernels/alu_bench.metal - instruction-rate probes (alu_bench.cpp): each thread runs ITER iterations of 8
// independent chains of one operation; the result is written so nothing is eliminated.
#define ITER 1024

kernel void a_fma32(device float* out [[buffer(0)]], constant float& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a0 = gid, a1 = gid + 1, a2 = gid + 2, a3 = gid + 3, a4 = gid + 4, a5 = gid + 5, a6 = gid + 6, a7 = gid + 7;
    for (int i = 0; i < ITER; ++i) {
        a0 = fma(a0, k, 0.5f); a1 = fma(a1, k, 0.5f); a2 = fma(a2, k, 0.5f); a3 = fma(a3, k, 0.5f);
        a4 = fma(a4, k, 0.5f); a5 = fma(a5, k, 0.5f); a6 = fma(a6, k, 0.5f); a7 = fma(a7, k, 0.5f);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

kernel void a_fma16(device float* out [[buffer(0)]], constant float& kf [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    const half k = half(kf);
    half a0 = gid & 7, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (int i = 0; i < ITER; ++i) {
        a0 = fma(a0, k, half(0.5)); a1 = fma(a1, k, half(0.5)); a2 = fma(a2, k, half(0.5)); a3 = fma(a3, k, half(0.5));
        a4 = fma(a4, k, half(0.5)); a5 = fma(a5, k, half(0.5)); a6 = fma(a6, k, half(0.5)); a7 = fma(a7, k, half(0.5));
    }
    out[gid] = float(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
}

// float accumulator, half operands (does the conversion come free?)
kernel void a_fmamix(device float* out [[buffer(0)]], constant float& kf [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    half h0 = half(gid & 7), h1 = h0 + half(1);
    float a0 = gid, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (int i = 0; i < ITER; ++i) {
        a0 = fma(float(h0), kf, a0); a1 = fma(float(h1), kf, a1); a2 = fma(float(h0), kf, a2); a3 = fma(float(h1), kf, a3);
        a4 = fma(float(h0), kf, a4); a5 = fma(float(h1), kf, a5); a6 = fma(float(h0), kf, a6); a7 = fma(float(h1), kf, a7);
        h0 = h0 * half(0.999); h1 = h1 * half(0.999);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

kernel void a_iand(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    uint a0 = gid, a1 = gid + 1, a2 = gid + 2, a3 = gid + 3, a4 = gid + 4, a5 = gid + 5, a6 = gid + 6, a7 = gid + 7;
    for (int i = 0; i < ITER; ++i) {
        a0 = (a0 ^ k) + 1; a1 = (a1 ^ k) + 1; a2 = (a2 ^ k) + 1; a3 = (a3 ^ k) + 1;
        a4 = (a4 ^ k) + 1; a5 = (a5 ^ k) + 1; a6 = (a6 ^ k) + 1; a7 = (a7 ^ k) + 1;
    }
    out[gid] = float(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
}

kernel void a_shift(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    uint a0 = gid, a1 = gid + 1, a2 = gid + 2, a3 = gid + 3, a4 = gid + 4, a5 = gid + 5, a6 = gid + 6, a7 = gid + 7;
    for (int i = 0; i < ITER; ++i) {
        a0 = (a0 >> k) | k; a1 = (a1 >> k) | k; a2 = (a2 >> k) | k; a3 = (a3 >> k) | k;
        a4 = (a4 >> k) | k; a5 = (a5 >> k) | k; a6 = (a6 >> k) | k; a7 = (a7 >> k) | k;
    }
    out[gid] = float(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
}

// int -> float conversion + fma
kernel void a_cvt(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    uint u = gid;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (int i = 0; i < ITER; ++i) {
        u += k;
        a0 += float(u & 3); a1 += float(u & 12); a2 += float(u & 48); a3 += float(u & 192);
        a4 += float(u & 768); a5 += float(u & 3072); a6 += float(u & 12288); a7 += float(u & 49152);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// threadgroup memory loads: random-ish 8-byte reads from a 4 KB table
kernel void a_tgload(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], uint gid [[thread_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint2 tab[512];
    for (int i = tid; i < 512; i += 256) tab[i] = uint2(i, i * 3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint u = gid * 2654435761u;
    uint2 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < ITER; ++i) {
        s0 += tab[(u >> 3) & 511]; s1 += tab[(u >> 7) & 511]; s2 += tab[(u >> 11) & 511]; s3 += tab[(u >> 15) & 511];
        u += k;
    }
    out[gid] = float(s0.x + s1.x + s2.x + s3.x + s0.y + s1.y + s2.y + s3.y);
}

// select (sign application)
kernel void a_sel(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    uint u = gid;
    float a0 = 1, a1 = 2, a2 = 3, a3 = 4, a4 = 5, a5 = 6, a6 = 7, a7 = 8;
    for (int i = 0; i < ITER; ++i) {
        u ^= k;
        a0 = (u & 1) ? -a0 : a0; a1 = (u & 2) ? -a1 : a1; a2 = (u & 4) ? -a2 : a2; a3 = (u & 8) ? -a3 : a3;
        a4 = (u & 16) ? -a4 : a4; a5 = (u & 32) ? -a5 : a5; a6 = (u & 64) ? -a6 : a6; a7 = (u & 128) ? -a7 : a7;
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// dot-shaped: acc_i += q_i * x_i with loop-carried accumulators only (q, x vary slowly)
kernel void a_dot32(device float* out [[buffer(0)]], constant float& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float q[8], x[8], a[8];
    for (int j = 0; j < 8; ++j) { q[j] = float(gid + j); x[j] = k * j; a[j] = 0; }
    for (int i = 0; i < ITER; ++i) {
        for (int j = 0; j < 8; ++j) a[j] = fma(q[j], x[j], a[j]);
        q[0] += 1.0f;
    }
    float s = 0;
    for (int j = 0; j < 8; ++j) s += a[j];
    out[gid] = s;
}
kernel void a_dot16(device float* out [[buffer(0)]], constant float& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    half q[8], x[8], a[8];
    for (int j = 0; j < 8; ++j) { q[j] = half(gid + j); x[j] = half(k * j); a[j] = 0; }
    for (int i = 0; i < ITER; ++i) {
        for (int j = 0; j < 8; ++j) a[j] = fma(q[j], x[j], a[j]);
        q[0] += half(1.0f);
    }
    float s = 0;
    for (int j = 0; j < 8; ++j) s += float(a[j]);
    out[gid] = s;
}
kernel void a_dotmix(device float* out [[buffer(0)]], constant float& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    half q[8];
    float x[8], a[8];
    for (int j = 0; j < 8; ++j) { q[j] = half(gid + j); x[j] = k * j; a[j] = 0; }
    for (int i = 0; i < ITER; ++i) {
        for (int j = 0; j < 8; ++j) a[j] = fma(float(q[j]), x[j], a[j]);
        q[0] += half(1.0f);
    }
    float s = 0;
    for (int j = 0; j < 8; ++j) s += a[j];
    out[gid] = s;
}
kernel void a_fma32x16(device float* out [[buffer(0)]], constant float& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a[16];
    for (int j = 0; j < 16; ++j) a[j] = gid + j;
    for (int i = 0; i < ITER / 2; ++i)
        for (int j = 0; j < 16; ++j) a[j] = fma(a[j], k, 0.5f);
    float s = 0;
    for (int j = 0; j < 16; ++j) s += a[j];
    out[gid] = s;
}

// 8-byte gathers from a 4 KB table in device memory (L1-resident) and in the constant address space
kernel void a_devload(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], device const uint2* tab [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint u = gid * 2654435761u;
    uint2 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < ITER; ++i) {
        s0 += tab[(u >> 3) & 511]; s1 += tab[(u >> 7) & 511]; s2 += tab[(u >> 11) & 511]; s3 += tab[(u >> 15) & 511];
        u += k;
    }
    out[gid] = float(s0.x + s1.x + s2.x + s3.x + s0.y + s1.y + s2.y + s3.y);
}
kernel void a_cstload(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], constant uint2* tab [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint u = gid * 2654435761u;
    uint2 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < ITER; ++i) {
        s0 += tab[(u >> 3) & 511]; s1 += tab[(u >> 7) & 511]; s2 += tab[(u >> 11) & 511]; s3 += tab[(u >> 15) & 511];
        u += k;
    }
    out[gid] = float(s0.x + s1.x + s2.x + s3.x + s0.y + s1.y + s2.y + s3.y);
}
// 4-byte gathers from a 1 KB threadgroup table (IQ3_XXS's grid)
kernel void a_tgload4(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], uint gid [[thread_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint tab[256];
    for (int i = tid; i < 256; i += 256) tab[i] = i * 3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint u = gid * 2654435761u;
    uint s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < ITER; ++i) {
        s0 += tab[(u >> 3) & 255]; s1 += tab[(u >> 7) & 255]; s2 += tab[(u >> 11) & 255]; s3 += tab[(u >> 15) & 255];
        u += k;
    }
    out[gid] = float(s0 + s1 + s2 + s3);
}
// popcount parity (computed ksigns)
kernel void a_popc(device float* out [[buffer(0)]], constant uint& k [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    uint u = gid, s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < ITER; ++i) {
        s0 += popcount(u & 127); s1 += popcount((u >> 7) & 127); s2 += popcount((u >> 14) & 127); s3 += popcount((u >> 21) & 127);
        u += k;
    }
    out[gid] = float(s0 + s1 + s2 + s3);
}
