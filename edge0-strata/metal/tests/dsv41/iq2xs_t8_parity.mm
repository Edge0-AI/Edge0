// iq2xs_t8_probe.mm - the IQ2_XS expert kernel, tiled over the engine's 8-token window (24 GB of the 2.25bpw set).
//
// Block (block_iq2_xs, 74 B): f16 d, 32 uint16 qs, 8 bytes of scales.  Per group of 32 weights:
//   db0 = d*(0.5 + (scales[ib] & 0xf))*0.25, db1 = d*(0.5 + (scales[ib] >> 4))*0.25
//   for l in 0..3: grid = iq2xs_grid[qs[4ib+l] & 511], signs = ksigns_iq2xs[qs[4ib+l] >> 9], scale = db[l/2]
// (ggml-quants.c dequantize_row_iq2_xs, pinned 3cf03257f.)  Decoded once per window, applied to 8 activations.
//   clang++ -O2 -fobjc-arc -framework Metal -framework Foundation -o iq2xs_t8_probe iq2xs_t8_probe.mm && ./iq2xs_t8_probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include <mach/mach_time.h>

static std::vector<uint64_t> table(const std::string& name, size_t expect) {
    std::ifstream f("ggml_grids.h");
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::regex re("constant \\w+ " + name + "\\[[^\\]]*\\] = \\{([^}]*)\\}");
    std::smatch m;
    if (!std::regex_search(all, m, re)) { fprintf(stderr, "no table %s\n", name.c_str()); exit(2); }
    std::vector<uint64_t> out;
    const std::string body = m[1];
    size_t i = 0;
    while (i < body.size() && out.size() < expect) {
        while (i < body.size() && !isalnum((unsigned char) body[i])) ++i;
        size_t j = i;
        while (j < body.size() && (isalnum((unsigned char) body[j]) || body[j] == 'x')) ++j;
        if (j > i) out.push_back(strtoull(body.substr(i, j - i).c_str(), nullptr, 0));
        i = j;
    }
    if (out.size() != expect) { fprintf(stderr, "%s: %zu of %zu\n", name.c_str(), out.size(), expect); exit(2); }
    return out;
}

int main() {
    @autoreleasepool {
        const int n = 65536, k = 5120, T = 8;
        const int nb = k / 256;
        const size_t bs = 74;
        auto grid = table("iq2xs_grid", 512);
        auto signs_t = table("ksigns_iq2xs", 128);

        std::vector<uint8_t> w((size_t) n * nb * bs);
        srand(9);
        for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (rand() & 0xff);
        for (int r = 0; r < n; ++r)
            for (int b = 0; b < nb; ++b) {
                _Float16 d = (_Float16) (((rand() % 900) + 100) / 8000.0f);
                memcpy(&w[((size_t) r * nb + b) * bs], &d, 2);
            }
        std::vector<float> x((size_t) T * k), y((size_t) T * n);
        for (size_t i = 0; i < x.size(); ++i) x[i] = ((rand() % 20001) - 10000) / 8192.0f;

        std::string grids;
        {
            std::ifstream f("ggml_grids.h");
            std::string line;
            while (std::getline(f, line)) {
                if (line.rfind("#pragma once", 0) == 0 || line.rfind("//", 0) == 0 || line.empty()) continue;
                grids += line + "\n";
            }
        }
        std::string src = "#include <metal_stdlib>\nusing namespace metal;\n" + grids + R"MSL(
kernel void iq2xs_t8(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* y [[buffer(2)]], constant uint& k [[buffer(3)]], constant uint& n [[buffer(4)]],
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint r = gid / 32u;
    if (r >= n) return;
    const uint nb = k / 256u;
    const device uchar* base = w + (size_t) r * nb * 74u;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = base + (size_t) b * 74u;
        const float d = float(as_type<half>(*(const device ushort*) blk));
        const device ushort* qs = (const device ushort*) (blk + 2);
        const device uchar* sc = blk + 2 + 64;
        for (int ib = 0; ib < 8; ++ib) {
            const float db0 = d * (0.5f + float(sc[ib] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + float(sc[ib] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint q = uint(qs[4 * ib + l]);
                const ulong g = iq2xs_grid[q & 511u];
                const uint sg = uint(ksigns_iq2xs[q >> 9]);
                const float db = (l < 2) ? db0 : db1;
                const uint xoff = (uint) ib * 32u + (uint) l * 8u;
                float wv[8];
                for (int j = 0; j < 8; ++j) {
                    wv[j] = db * float((g >> (8 * j)) & 0xff) * ((sg & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                for (uint t = 0; t < 8u; ++t) {
                    const device float* xt = x + (size_t) t * k + xoff;
                    float s = 0.0f;
                    for (int j = 0; j < 8; ++j) s += wv[j] * xt[j];
                    acc[t] += s;
                }
            }
        }
    }
    for (uint t = 0; t < 8u; ++t) {
        float a = acc[t];
        a += simd_shuffle_xor(a, 16);
        a += simd_shuffle_xor(a, 8);
        a += simd_shuffle_xor(a, 4);
        a += simd_shuffle_xor(a, 2);
        a += simd_shuffle_xor(a, 1);
        if (lane == 0) y[(size_t) t * n + r] = a;
    }
}
)MSL";

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:nil error:&err];
        if (!lib) { printf("MSL compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"iq2xs_t8"] error:&err];
        if (!pso) { printf("pipeline failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLBuffer> bw = [dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:x.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [dev newBufferWithLength:y.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        unsigned uk = k, un = n;
        double best = 1e9;
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        for (int it = 0; it < 5; ++it) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:pso];
            [ce setBuffer:bw offset:0 atIndex:0];
            [ce setBuffer:bx offset:0 atIndex:1];
            [ce setBuffer:by offset:0 atIndex:2];
            [ce setBytes:&uk length:4 atIndex:3];
            [ce setBytes:&un length:4 atIndex:4];
            [ce dispatchThreads:MTLSizeMake((NSUInteger) n * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [ce endEncoding];
            const uint64_t t0 = mach_absolute_time();
            [cb commit];
            [cb waitUntilCompleted];
            const double dt = (double) (mach_absolute_time() - t0) * tb.numer / tb.denom / 1e9;
            if (dt < best) best = dt;
        }
        printf("IQ2_XS window of 8: %.2f GB/s effective (%.2f raw), %.3f ms for %.1f MB\n", 8.0 * w.size() / best / 1e9,
               (double) w.size() / best / 1e9, best * 1e3, (double) w.size() / 1e6);

        const float* gy = (const float*) [by contents];
        const uint8_t* kmask = (const uint8_t*) "\x01\x02\x04\x08\x10\x20\x40\x80";
        double num = 0, den = 0;
        for (int r = 0; r < n; r += 61) {
            for (int t = 0; t < T; ++t) {
                double acc = 0;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* blk = &w[((size_t) r * nb + b) * bs];
                    _Float16 dh; memcpy(&dh, blk, 2);
                    const double d = (double) (float) dh;
                    const uint8_t* sc = blk + 2 + 64;
                    const uint8_t* qb = blk + 2;
                    for (int ib = 0; ib < 8; ++ib) {
                        const double db0 = d * (0.5 + (double) (sc[ib] & 0xf)) * 0.25;
                        const double db1 = d * (0.5 + (double) (sc[ib] >> 4)) * 0.25;
                        for (int l = 0; l < 4; ++l) {
                            uint16_t q16;
                            memcpy(&q16, qb + 2 * (4 * ib + l), 2);
                            const uint64_t g = grid[q16 & 511u];
                            const uint32_t sg = (uint32_t) signs_t[(q16 >> 9) & 127u];
                            const double db = (l < 2) ? db0 : db1;
                            const int xoff = ib * 32 + l * 8;
                            for (int j = 0; j < 8; ++j) {
                                const double wv = db * (double) ((g >> (8 * j)) & 0xff) * ((sg & kmask[j]) ? -1.0 : 1.0);
                                acc += wv * (double) x[(size_t) t * k + xoff + j];
                            }
                        }
                    }
                }
                const double diff = (double) gy[(size_t) t * n + r] - acc;
                num += diff * diff;
                den += acc * acc;
            }
        }
        const double l2 = sqrt(num / den);
        printf("relative L2 %.3e (sampled rows)\n", l2);
        printf("%s\n", l2 < 1e-5 ? "PASS" : "FAIL");
    }
    return 0;
}
