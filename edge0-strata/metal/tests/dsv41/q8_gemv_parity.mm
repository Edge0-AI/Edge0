// q8gemv_probe.mm - the quantised-weight GEMV in MSL: Q8_0 weights times F32 activations (ds4_gguf.cu's Q8 path).
//
// Q8_0 block: 32 int8 values + one f16 scale d, 34 bytes.  y[r] = sum over blocks of d * sum_i q[i] * x[i].  This is the
// simplest quantised-weight kernel and the one every Q8 tensor in the pack uses; IQ2/IQ3 use the same shape with a grid
// table instead of a plain int8 array, so this validates the code path before the harder formats.
//   clang++ -O2 -fobjc-arc -framework Metal -framework Foundation -o q8gemv_probe q8gemv_probe.mm && ./q8gemv_probe
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

// one warp a row, lane-strided over the blocks
kernel void ds4_q8_gemv(device const char* q [[buffer(0)]], device const half* d [[buffer(1)]],
                        device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                        constant uint& k [[buffer(4)]], constant uint& n [[buffer(5)]],
                        uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint r = gid / 32u;
    if (r >= n) return;
    const uint nb = k / 32u;                       // blocks a row
    const device char* qr = q + (size_t) r * k;    // k int8 values, one a float
    const device half* dr = d + (size_t) r * nb;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        float s = 0.0f;
        for (uint i = 0; i < 32u; ++i) s += float(qr[b * 32u + i]) * x[b * 32u + i];
        acc += s * float(dr[b]);
    }
    acc += simd_shuffle_xor(acc, 16);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) y[r] = acc;
}
)MSL";

int main() {
    @autoreleasepool {
        const int n = 16384, k = 5120;
        const int nb = k / 32;
        std::vector<signed char> q((size_t) n * k);
        std::vector<float> dv((size_t) n * nb), x(k), y(n);
        srand(5);
        for (size_t i = 0; i < q.size(); ++i) q[i] = (signed char) ((rand() % 255) - 127);
        for (size_t i = 0; i < dv.size(); ++i) dv[i] = ((rand() % 1000) + 1) / 4000.0f;
        for (int i = 0; i < k; ++i) x[i] = ((rand() % 20001) - 10000) / 8192.0f;

        // the f16 scales as half bits
        std::vector<unsigned short> dh(dv.size());
        for (size_t i = 0; i < dv.size(); ++i) {
            _Float16 h = (_Float16) dv[i];
            memcpy(&dh[i], &h, 2);
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
        if (!lib) { printf("MSL compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ds4_q8_gemv"] error:&err];
        if (!pso) { printf("pipeline failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLBuffer> bq = [dev newBufferWithBytes:q.data() length:q.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bd = [dev newBufferWithBytes:dh.data() length:dh.size() * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:k * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> qq = [dev newCommandQueue];
        unsigned uk = k, un = n;
        double best = 1e9;
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        for (int it = 0; it < 6; ++it) {
            id<MTLCommandBuffer> cb = [qq commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:pso];
            [ce setBuffer:bq offset:0 atIndex:0];
            [ce setBuffer:bd offset:0 atIndex:1];
            [ce setBuffer:bx offset:0 atIndex:2];
            [ce setBuffer:by offset:0 atIndex:3];
            [ce setBytes:&uk length:4 atIndex:4];
            [ce setBytes:&un length:4 atIndex:5];
            [ce dispatchThreads:MTLSizeMake((NSUInteger) n * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [ce endEncoding];
            const uint64_t t0 = mach_absolute_time();
            [cb commit];
            [cb waitUntilCompleted];
            const double dt = (double) (mach_absolute_time() - t0) * tb.numer / tb.denom / 1e9;
            if (dt < best) best = dt;
        }
        const double bytes = (double) q.size() + (double) dh.size() * 2.0;
        printf("throughput: %.2f GB/s, %.2f GFLOP/s (%.3f ms for %.1f MB)\n", bytes / best / 1e9, 2.0 * n * k / best / 1e9,
               best * 1e3, bytes / 1e6);

        const float* gy = (const float*) [by contents];
        double num = 0, den = 0;
        for (int r = 0; r < n; ++r) {
            double acc = 0;
            for (int b = 0; b < nb; ++b) {
                double s = 0;
                for (int i = 0; i < 32; ++i) s += (double) q[(size_t) r * k + b * 32 + i] * (double) x[b * 32 + i];
                acc += s * (double) (float) (_Float16) dv[(size_t) r * nb + b];   // the kernel gets the half-rounded scale
            }
            const double diff = (double) gy[r] - acc;
            num += diff * diff;
            den += acc * acc;
        }
        const double l2 = sqrt(num / den);
        printf("relative L2 %.3e\n", l2);
        printf("%s\n", l2 < 1e-5 ? "PASS" : "FAIL");
    }
    return 0;
}
