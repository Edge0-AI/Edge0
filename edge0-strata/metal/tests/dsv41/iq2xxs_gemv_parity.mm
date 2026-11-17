// iq2xxs_probe.mm - the IQ2_XXS expert GEMV in MSL, the 2.25bpw set's dominant expert format (49 GB of the 149 GB).
//
// Decode (ggml-quants.c dequantize_row_iq2_xxs, pinned 3cf03257f): a block is 66 bytes - an f16 d and 32 bytes of qs.  Eight
// groups of 32 weights: aux32[0..1] are the group's 8 bytes; db = d * (0.5 + (aux32[1] >> 28)) * 0.25; each of the group's four
// sub-blocks takes grid = iq2xxs_grid[aux8[l]] (8 bytes) and signs = ksigns_iq2xs[(aux32[1] >> 7l) & 127], giving
// y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1).
//
// The grid tables are read from the generated MSL header so the CPU reference uses exactly the same numbers as the GPU.
//   clang++ -O2 -fobjc-arc -framework Metal -framework Foundation -o iq2xxs_probe iq2xxs_probe.mm && ./iq2xxs_probe
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
    std::string body = m[1];
    size_t i = 0;
    while (i < body.size()) {
        while (i < body.size() && !isdigit((unsigned char) body[i])) ++i;
        if (i >= body.size()) break;
        size_t j = i;
        while (j < body.size() && (isdigit((unsigned char) body[j]) || body[j] == 'x' || isxdigit((unsigned char) body[j]))) ++j;
        out.push_back(strtoull(body.substr(i, j - i).c_str(), nullptr, 0));
        i = j;
        if (out.size() > expect) break;
    }
    if (out.size() != expect) { fprintf(stderr, "table %s: %zu of %zu\n", name.c_str(), out.size(), expect); exit(2); }
    return out;
}

int main() {
    @autoreleasepool {
        const int n = 65536, k = 5120;               // rows and k
        const int nb = k / 256;                     // IQ2_XXS blocks a row
        const size_t bs = 66;                       // bytes a block
        auto grid = load_msl_table("ggml_grids.h", "iq2xxs_grid", 256);
        auto signs_t = load_msl_table("ggml_grids.h", "ksigns_iq2xs", 128);

        // a synthetic pack of blocks (the decoder does not care where the bytes come from)
        std::vector<uint8_t> w((size_t) n * nb * bs);
        srand(3);
        for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (rand() & 0xff);
        for (int r = 0; r < n; ++r)
            for (int b = 0; b < nb; ++b) {
                // a sane f16 scale in the block's first two bytes
                _Float16 d = (_Float16) (((rand() % 900) + 100) / 8000.0f);
                memcpy(&w[((size_t) r * nb + b) * bs], &d, 2);
            }
        std::vector<float> x(k), y(n);
        for (int i = 0; i < k; ++i) x[i] = ((rand() % 20001) - 10000) / 8192.0f;

        // pull the grid table into the MSL by textual include at compile time
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
kernel void iq2xxs_gemv(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant uint& k [[buffer(3)]], constant uint& n [[buffer(4)]],
                        uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint r = gid / 32u;
    if (r >= n) return;
    const uint nb = k / 256u;
    const device uchar* base = w + (size_t) r * nb * 66u;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = base + (size_t) b * 66u;
        const float d = float(as_type<half>(*(const device ushort*) blk));
        for (int ib = 0; ib < 8; ++ib) {
            // groups advance 4 bytes and read 8, so an 8-byte vector load is misaligned on odd groups: read bytes
            const uint aux0 = uint(blk[2 + 4 * ib]) | (uint(blk[3 + 4 * ib]) << 8) | (uint(blk[4 + 4 * ib]) << 16) | (uint(blk[5 + 4 * ib]) << 24);
            const uint aux1 = uint(blk[6 + 4 * ib]) | (uint(blk[7 + 4 * ib]) << 8) | (uint(blk[8 + 4 * ib]) << 16) | (uint(blk[9 + 4 * ib]) << 24);
            const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const ulong g = iq2xxs_grid[(aux0 >> (8 * l)) & 0xff];
                const uint sg = uint(ksigns_iq2xs[(aux1 >> (7 * l)) & 127]);
                const uint xoff = (uint) ib * 32u + (uint) l * 8u;
                for (int j = 0; j < 8; ++j) {
                    const float wv = db * float((g >> (8 * j)) & 0xff) * ((sg & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    acc += wv * x[xoff + j];
                }
            }
        }
    }
    acc += simd_shuffle_xor(acc, 16);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0) y[r] = acc;
}
)MSL";
        const uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        // kmask_iq2xs comes from ggml_grids.h
        std::string src_with_km = src;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src_with_km.c_str()] options:nil error:&err];
        if (!lib) { printf("MSL compile failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"iq2xxs_gemv"] error:&err];
        if (!pso) { printf("pipeline failed: %s\n", [[err localizedDescription] UTF8String]); return 1; }

        id<MTLBuffer> bw = [dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:k * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
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
        printf("throughput: %.2f GB/s (%.3f ms for %.1f MB)\n", (double) w.size() / best / 1e9, best * 1e3,
               (double) w.size() / 1e6);

        // the CPU reference: the same decode, from the same tables
        const float* gy = (const float*) [by contents];
        double num = 0, den = 0;
        for (int r = 0; r < n; ++r) {
            double acc = 0;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* blk = &w[((size_t) r * nb + b) * bs];
                _Float16 dh;
                memcpy(&dh, blk, 2);
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
                            acc += wv * (double) x[xoff + j];
                        }
                    }
                }
            }
            const double diff = (double) gy[r] - acc;
            num += diff * diff;
            den += acc * acc;
        }
        const double l2 = sqrt(num / den);
        printf("relative L2 %.3e\n", l2);
        printf("%s\n", l2 < 1e-4 ? "PASS" : "FAIL");
    }
    return 0;
}
