// src/kernels/latent_source_check.cpp - FileExpertSource/ExpertLayout on the 48-layer latent A pack.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <latent_a_pack> [layer0_native.bin]\n", argv[0]); return 2; }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(argv[1], 48, 512, err)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1;
    }
    strata::core::FileExpertSource src;
    if (!src.open(argv[1], 48, 512, err)) {
        std::fprintf(stderr, "source: %s\n", err.c_str()); return 1;
    }
    const auto& L = strata::kernels::cpu::expert_layout();
    std::printf("mapped=%d blobs=%lld max_blob=%llu\n", (int) src.mapped(), (long long) src.blobs(),
                (unsigned long long) L.max_blob);
    int bad = 0;
    for (int64_t l : {0LL, 1LL, 23LL, 47LL}) {
        for (int e : {0, 1, 256, 511}) {
            const uint8_t* p = src.blob(l, e);
            if (!p) { std::printf("null layer %lld expert %d\n", (long long) l, e); ++bad; continue; }
            (void) p;
        }
    }
    if (argc > 2) {
        std::ifstream f(argv[2], std::ios::binary);
        std::vector<uint8_t> want(L.blob_bytes(0));
        f.seekg((std::streamoff) 256 * L.blob_bytes(0));
        f.read((char*) want.data(), (std::streamsize) want.size());
        const uint8_t* p = src.blob(0, 256);
        const bool same = p && std::memcmp(p, want.data(), want.size()) == 0;
        std::printf("layer0 expert256 byte-equal to single-layer pack: %s\n", same ? "yes" : "NO");
        bad += same ? 0 : 1;
    }
    std::printf("RESULT: %s\n", bad == 0 ? "ok" : "BAD");
    return bad == 0 ? 0 : 1;
}
