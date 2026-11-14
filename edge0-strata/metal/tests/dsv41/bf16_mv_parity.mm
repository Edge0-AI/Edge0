// bf16mv_probe.mm - the second dsv41 kernel in MSL: the BF16 matrix-vector product (ds4_gguf.cu:1043's mx_mv_kernel /
// bf16_mv_kernel, single-job case).
//
// CUDA's layout: one warp per row, lane i handles k = i, i + 32, ... in BF16 weights times F32 activations, accumulated in
// F32.  This compiles the MSL version at run time, runs it on the GPU and compares against the same arithmetic on the CPU.
//   clang++ -O2 -fobjc-arc -framework Metal -framework Foundation -o bf16mv_probe bf16mv_probe.mm && ./bf16mv_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <mach/mach_time.h>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// y[r] = sum_k bf16_to_float(w[r*k + kk]) * x[kk]   (one warp a row)
kernel void ds4_bf16_mv(device const ushort* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant uint& k [[buffer(3)]], constant uint& n [[buffer(4)]],
                        uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint rows_per_group = 1u;                 // one warp covers one row
    const uint r = gid / 32u * rows_per_group;      // the row this warp owns
    if (r >= n) return;
    float acc = 0.0f;
    for (uint kk = lane; kk < k; kk += 32u) {
        const uint bits = uint(w[r * k + kk]) << 16;   // BF16 -> F32 is the top half
        const float wf = as_type<float>(bits);
        acc += wf * x[kk];
    }
    // warp reduce
    acc += simd_shuffle_xor(acc, 16);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) y[r] = acc;
}
)MSL";

static float bf16_to_f32(unsigned short b) {
    unsigned bits = (unsigned) b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
static unsigned short f32_to_bf16(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (unsigned short) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

int main() {
    @autoreleasepool {
        const int n = 2048, k = 5120;
        std::vector<unsigned short> w((size_t) n * k);
        std::vector<float> x(k), y(n);
        srand(11);
        for (size_t i = 0; i < w.size(); ++i) w[i] = f32_to_bf16(((rand() % 20001) - 10000) / 4096.0f);
        for (int i = 0; i < k; ++i) x[i] = ((rand() % 20001) - 10000) / 8192.0f;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
        if (!lib) { printf("MSL compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ds4_bf16_mv"] error:&err];
        if (!pso) { printf("pipeline failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }

        id<MTLBuffer> bw = [dev newBufferWithBytes:w.data() length:w.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:k * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pso];
        [ce setBuffer:bw offset:0 atIndex:0];
        [ce setBuffer:bx offset:0 atIndex:1];
        [ce setBuffer:by offset:0 atIndex:2];
        unsigned uk = k, un = n;
        [ce setBytes:&uk length:4 atIndex:3];
        [ce setBytes:&un length:4 atIndex:4];
        [ce dispatchThreads:MTLSizeMake((NSUInteger) n * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { printf("dispatch error: %s\n", [[cb.error localizedDescription] UTF8String]); return 1; }

        // throughput: 10 more dispatches, best of them
        double best = 1e9;
        for (int it = 0; it < 10; ++it) {
            id<MTLCommandBuffer> c2 = [q commandBuffer];
            id<MTLComputeCommandEncoder> e2 = [c2 computeCommandEncoder];
            [e2 setComputePipelineState:pso];
            [e2 setBuffer:bw offset:0 atIndex:0];
            [e2 setBuffer:bx offset:0 atIndex:1];
            [e2 setBuffer:by offset:0 atIndex:2];
            [e2 setBytes:&uk length:4 atIndex:3];
            [e2 setBytes:&un length:4 atIndex:4];
            [e2 dispatchThreads:MTLSizeMake((NSUInteger) n * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e2 endEncoding];
            mach_timebase_info_data_t tb; mach_timebase_info(&tb);
            const uint64_t t0 = mach_absolute_time();
            [c2 commit];
            [c2 waitUntilCompleted];
            const double dt = (double) (mach_absolute_time() - t0) * tb.numer / tb.denom / 1e9;
            if (dt < best) best = dt;
        }
        {
            const double bytes = (double) n * k * 2.0, flops = 2.0 * n * k;
            printf("throughput: %.2f GB/s, %.1f GFLOP/s (%.3f ms for 21.0 MB)\n", bytes / best / 1e9,
                   flops / best / 1e9, best * 1e3);
        }
        const float* gy = (const float*) [by contents];
        // the standard measure for a dot product: relative L2 over the whole result, not per row (a 5120-term float sum
        // cancels, so per-row relative error is ~1e-3 even when the kernels agree to float precision)
        double num = 0, den = 0, wrel = 0;
        for (int r = 0; r < n; ++r) {
            double acc = 0;
            for (int kk = 0; kk < k; ++kk) acc += (double) bf16_to_f32(w[(size_t) r * k + kk]) * (double) x[kk];
            const double d = (double) gy[r] - acc;
            num += d * d;
            den += acc * acc;
            const double rel = fabs(d) / (fabs(acc) > 1e-9 ? fabs(acc) : 1.0);
            wrel = rel > wrel ? rel : wrel;
        }
        const double l2 = sqrt(num / den);
        printf("rows %d k %d: relative L2 %.3e, worst row %.3e\n", n, k, l2, wrel);
        printf("%s\n", l2 < 1e-5 ? "PASS" : "FAIL");
    }
    return 0;
}
