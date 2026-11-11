// src/kernels/latent_layout_check.cpp - ExpertLayout v6: 48 x 512 latent A blobs.
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <latent_a_pack>\n", argv[0]); return 2; }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(argv[1], 48, 512, err)) {
        std::fprintf(stderr, "layout load failed: %s\n", err.c_str());
        return 1;
    }
    const auto& L = strata::kernels::cpu::expert_layout();
    std::printf("native=%d layers=%lld experts=%lld max_blob=%llu total=%llu\n", (int) L.native,
                (long long) L.n_layers, (long long) L.n_expert, (unsigned long long) L.max_blob,
                (unsigned long long) L.total);
    if (L.fmt.empty()) { std::fprintf(stderr, "no fmt\n"); return 1; }
    const auto& f = L.fmt[0];
    std::printf("layer0 embd=%lld ff=%lld gu=%d d=%d bytes=%llu blob0=%llu\n", (long long) f.n_embd,
                (long long) f.n_ff, f.gu_type, f.d_type, (unsigned long long) f.bytes,
                (unsigned long long) L.blob_bytes(0));
    const uint64_t want_blob = 1280ull * 1008 + 1792ull * 360 + 16;
    const bool ok = L.native && L.n_layers == 48 && L.n_expert == 512 && L.max_blob == want_blob &&
                    L.blob_bytes(0) == want_blob && f.n_embd == 1792 && f.n_ff == 640 &&
                    L.total == 48ull * 512ull * want_blob;
    std::printf("RESULT: %s (want blob %llu, total %llu)\n", ok ? "ok" : "BAD",
                (unsigned long long) want_blob, (unsigned long long) (48ull * 512ull * want_blob));
    return ok ? 0 : 1;
}
