// iq2xxs_t8_probe.mm - the tiled IQ2_XXS expert kernel: one weight decode feeding 8 tokens (the engine's 8-token window).
//
// The warp-per-row kernel measured 87.7 GB/s, which caps decode near 38 tok/s; the decode is the cost, so it must be paid once
// per window instead of once per token.  Here each lane decodes its blocks ONCE and applies them to 8 activations, keeping eight
// accumulators.  Success is a per-token effective rate well above the single-token one.
//   clang++ -O2 -fobjc-arc -framework Metal -framework Foundation -o iq2xxs_t8_probe iq2xxs_t8_probe.mm && ./iq2xxs_t8_probe
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

static std::vector<uint64_t> load_msl_table(const std::string& path, const std::string& name, size_t expect) {
    std::ifstream f(path);
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::regex re("constant \\w+ " + name + "\\[[^\\]]*\\] = \\{([^}]*)\\}");
    std::smatch m;
    std::vector<uint64_t> out;
    if (!std::regex_search(all, m, re)) { fprintf(stderr, "table %s not found\n", name.c_str()); exit(2); }
    const std::string body = m[1];
    size_t i = 0;
    while (i < body.size() && out.size() < expect) {
        while (i < body.size() && !isalnum((unsigned char) body[i])) ++i;
        size_t j = i;
        while (j < body.size() && (isalnum((unsigned char) body[j]) || body[j] == 'x')) ++j;
        if (j > i) out.push_back(strtoull(body.substr(i, j - i).c_str(), nullptr, 0));
        i = j;
    }
    if (out.size() != expect) { fprintf(stderr, "table %s: %zu of %zu\n", name.c_str(), out.size(), expect); exit(2); }
    return out;
}

int main() {
    @autoreleasepool {
        const int n = 65536, k = 5120, T = 8;
        const int nb = k / 256;
        const size_t bs = 66;
        auto grid = load_msl_table("ggml_grids.h", "iq2xxs_grid", 256);
        auto signs_t = load_msl_table("ggml_grids.h", "ksigns_iq2xs", 128);

        std::vector<uint8_t> w((size_t) n * nb * bs);
        srand(3);
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
constant uint kTokens = 8u;

kernel void iq2xxs_t8(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]], constant uint& k [[buffer(3)]], constant uint& n [[buffer(4)]],
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint r = gid / 32u;
    if (r >= n) return;
    const uint nb = k / 256u;
    const device uchar* base = w + (size_t) r * nb * 66u;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = base + (size_t) b * 66u;
        const float d = float(as_type<half>(*(const device ushort*) blk));
        for (int ib = 0; ib < 8; ++ib) {
            const uint aux0 = uint(blk[2 + 4 * ib]) | (uint(blk[3 + 4 * ib]) << 8) | (uint(blk[4 + 4 * ib]) << 16) | (uint(blk[5 + 4 * ib]) << 24);
            const uint aux1 = uint(blk[6 + 4 * ib]) | (uint(blk[7 + 4 * ib]) << 8) | (uint(blk[8 + 4 * ib]) << 16) | (uint(blk[9 + 4 * ib]) << 24);
            const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const ulong g = iq2xxs_grid[(aux0 >> (8 * l)) & 0xff];
                const uint sg = uint(ksigns_iq2xs[(aux1 >> (7 * l)) & 127]);
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
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"iq2xxs_t8"] error:&err];
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
        // the window reads the weights once and produces 8 tokens of output
        printf("window of 8: %.2f GB/s effective (%.2f GB/s raw), %.3f ms for %.1f MB\n", 8.0 * w.size() / best / 1e9,
               (double) w.size() / best / 1e9, best * 1e3, (double) w.size() / 1e6);

        const float* gy = (const float*) [by contents];
        const uint8_t* kmask = (const uint8_t*) "\x01\x02\x04\x08\x10\x20\x40\x80";
        double num = 0, den = 0;
        for (int r = 0; r < n; r += 97) {                 // sampled: the reference is O(T * n * k)
            for (int t = 0; t < T; ++t) {
                double acc = 0;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* blk = &w[((size_t) r * nb + b) * bs];
                    _Float16 dh; memcpy(&dh, blk, 2);
                    const double d = (double) (float) dh;
                    for (int ib = 0; ib < 8; ++ib) {
                        uint32_t aux[2];
                        memcpy(aux, blk + 2 + 4 * ib, 8);
                        const double db = d * (0.5 + (double) (aux[1] >> 28)) * 0.25;
                        const uint8_t* a8 = (const uint8_t*) aux;
                        for (int l = 0; l < 4; ++l) {
                            const uint64_t g = grid[a8[l]];
                            const uint32_t sg = (uint32_t) signs_t[(aux[1] >> (7 * l)) & 127];
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
