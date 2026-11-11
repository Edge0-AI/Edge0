// tools/dsv41_lab/hip/wmma_probe.hip.cpp - run this FIRST on the gfx1201 box (one launch, no inputs): it checks the lane layout of
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 that wmma_emu.hpp and tqk_gemv_wmma.hip.cpp assume:
//   A[m][k]: lane l holds a[j] = A[l % 16][8 * (l / 16) + j];   B[k][n]: lane l holds b[j] = B[8 * (l / 16) + j][l % 16]
//   D[m][n]: lane l holds c[i] = D[8 * (l / 16) + i][l % 16]
// Probe 1: A[m][k] = m * 16 + k (exact in fp16), B = identity  =>  D = A, so lane l's c[i] must be (8 (l/16) + i) * 16 + (l % 16).
// Probe 2: A = identity, B[k][n] = k * 16 + n               =>  D = B, same expectation.
// A mismatch in probe 1 only means the A (or D) layout differs; in probe 2 only B (or D): the printout says which.
//   hipcc --offload-arch=gfx1201 -O2 wmma_probe.hip.cpp -o wmma_probe && HIP_VISIBLE_DEVICES=<card> ./wmma_probe
#include <hip/hip_runtime.h>
#include <cstdio>

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));

__global__ void probe(float* out, int which) {
    const int l = threadIdx.x;   // wave32
    half8_t a, b;
    for (int j = 0; j < 8; ++j) {
        const int m = l % 16, k = 8 * (l / 16) + j;   // the layout under test, for A
        const int k2 = 8 * (l / 16) + j, n = l % 16;  // for B
        a[j] = (_Float16) (which == 0 ? (float) (m * 16 + k) : (m == k ? 1.f : 0.f));
        b[j] = (_Float16) (which == 0 ? (k2 == n ? 1.f : 0.f) : (float) (k2 * 16 + n));
    }
    float8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; ++i) out[(which * 32 + l) * 8 + i] = c[i];
}

int main() {
    float* d; hipMalloc(&d, 2 * 32 * 8 * sizeof(float));
    probe<<<1, 32>>>(d, 0); probe<<<1, 32>>>(d, 1);
    float h[2 * 32 * 8]; hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost);
    int bad[2] = {0, 0};
    for (int w = 0; w < 2; ++w)
        for (int l = 0; l < 32; ++l)
            for (int i = 0; i < 8; ++i) bad[w] += h[(w * 32 + l) * 8 + i] != (float) ((8 * (l / 16) + i) * 16 + (l % 16));
    std::printf("probe 1 (A layout, D layout): %d of 256 values wrong\nprobe 2 (B layout, D layout): %d of 256 wrong\n%s\n", bad[0], bad[1],
                (bad[0] || bad[1]) ? "LAYOUT DIFFERS: fix wmma_emu.hpp first (the emulator is the specification), then the kernel" : "layout as assumed");
    return bad[0] || bad[1];
}
