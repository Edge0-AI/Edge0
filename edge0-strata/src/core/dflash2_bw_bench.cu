// read-bandwidth probe: sum of a 256 MB buffer with float4 loads, a few launch shapes
#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const float4* __restrict__ p, size_t n, float* out) {
    float s = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        float4 v = p[i];
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 1234.5f) out[0] = s;
}
__global__ void rd_unroll(const int4* __restrict__ p, size_t n, int* out) {
    int s = 0;
    size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        int4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        s += a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    if (s == 1234) out[0] = s;
}
int main() {
    const size_t bytes = 256u << 20, n = bytes / 16;
    void* p; float* o;
    cudaMalloc(&p, bytes); cudaMalloc(&o, 4);
    cudaMemset(p, 1, bytes);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int blocks : {112, 224, 448, 896, 3584, 14336}) {
        for (int k = 0; k < 2; ++k) {
            cudaEventRecord(a);
            for (int it = 0; it < 20; ++it) {
                if (k == 0) rd<<<blocks, 256>>>((const float4*) p, n, o);
                else rd_unroll<<<blocks, 256>>>((const int4*) p, n, (int*) o);
            }
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            printf("%s blocks %5d: %.1f GB/s\n", k == 0 ? "loop  " : "unroll", blocks, 20.0 * bytes / (ms / 1000) / 1e9);
        }
    }
    return 0;
}
