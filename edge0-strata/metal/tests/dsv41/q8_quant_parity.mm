// q8probe.mm - the first dsv41 kernel in MSL: the activation quantiser at ds4_gguf.cu:201.
//
// CUDA's q8 path: per 32 values, d = max|x| / 127 (a half), then q = round(x / d) clamped to [-128, 127], stored as int8
// with the f16 scale.  This compiles the MSL version at run time, runs it on the M5 Max GPU, and compares it against the
// same arithmetic in C++ - if the two agree, this function is ready to become a Pipeline in metal/src/engine.
//   clang++ -O2 -fobjc-arc -framework Metal -framework Foundation -o q8probe q8probe.mm && ./q8probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// one threadgroup of 32 threads per block of 32 values (CUDA's layout), one block per 32 inputs
kernel void q8_quant(device const float* x [[buffer(0)]], constant uint& ldx [[buffer(1)]],
                     device char* q [[buffer(2)]], device half* d [[buffer(3)]],
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint b = gid / 32u;   // the block this thread belongs to
    const int base = int(b) * int(ldx);
    float v = x[base + int(lane)];
    float amax = fabs(v);
    amax = max(amax, simd_shuffle_xor(amax, 16));
    amax = max(amax, simd_shuffle_xor(amax, 8));
    amax = max(amax, simd_shuffle_xor(amax, 4));
    amax = max(amax, simd_shuffle_xor(amax, 2));
    amax = max(amax, simd_shuffle_xor(amax, 1));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int iq = int(rint(v * inv));
    iq = clamp(iq, -128, 127);
    q[int(b) * 32 + int(lane)] = char(iq);
    if (lane == 0) d[int(b)] = half(scale);
}
)MSL";

int main() {
    @autoreleasepool {
        const int n = 32 * 4096;
        std::vector<float> x(n);
        srand(7);
        for (int i = 0; i < n; ++i) x[i] = ((rand() % 20001) - 10000) / 997.0f;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
        if (!lib) { printf("MSL compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"q8_quant"] error:&err];
        if (!pso) { printf("pipeline failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }

        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:n * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bd = [dev newBufferWithLength:(n / 32) * 2 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> qq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [qq commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pso];
        [ce setBuffer:bx offset:0 atIndex:0];
        unsigned ldx = 32;   // the stride between blocks of 32 values
        [ce setBytes:&ldx length:4 atIndex:1];
        [ce setBuffer:bq offset:0 atIndex:1 + 1];
        [ce setBuffer:bd offset:0 atIndex:2 + 1];
        [ce dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // the same arithmetic on the CPU
        const signed char* gq = (const signed char*) [bq contents];
        const _Float16* gd = (const _Float16*) [bd contents];
        // first block, side by side
        {
            float amax = 0; for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
            const float scale = amax / 127.0f, inv = scale > 0 ? 1.0f / scale : 0.0f;
            printf("block0: gpu d=%.6f cpu d=%.6f\n", (double) gd[0], (double) scale);
            printf("  gpu q:"); for (int i = 0; i < 8; ++i) printf(" %d", (int) gq[i]);
            printf("\n  cpu q:"); for (int i = 0; i < 8; ++i) printf(" %d", (int) rintf(x[i] * inv));
            printf("\nblock1: gpu d=%.6f  gpu q:", (double) gd[1]); for (int i = 32; i < 40; ++i) printf(" %d", (int) gq[i]);
            float am1 = 0; for (int i = 32; i < 64; ++i) am1 = fmaxf(am1, fabsf(x[i]));
            printf("\n  cpu d=%.6f  cpu q:", (double) (am1 / 127.0f)); for (int i = 32; i < 40; ++i) printf(" %d", (int) rintf(x[i] * (127.0f / am1)));
            printf("\n");
        }        int bad = 0; double worst = 0;
        for (int b = 0; b < n / 32; ++b) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
            const float scale = amax / 127.0f;
            const float inv = scale > 0 ? 1.0f / scale : 0.0f;
            const double dd = fabs((double) gd[b] - (double) scale) / (scale > 0 ? scale : 1);
            worst = fmax(worst, dd);
            for (int i = 0; i < 32; ++i) {
                int iq = (int) rintf(x[b * 32 + i] * inv);
                iq = iq < -128 ? -128 : (iq > 127 ? 127 : iq);
                if (iq != gq[b * 32 + i]) ++bad;
            }
        }
        printf("blocks %d, values %d\n", n / 32, n);
        printf("int8 differ: %d   worst scale rel err %.3e\n", bad, worst);
        printf("%s\n", bad == 0 && worst < 1e-3 ? "PASS" : "FAIL");
    }
    return 0;
}
