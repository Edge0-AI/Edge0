#include "strata/core/latent_dense.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
bool cmp_layer(const strata::core::LatentDense& d, const char* test, int layer, int T, int K, int E, int H) {
    std::ifstream f(test, std::ios::binary);
    if (!f) { std::printf("cannot open %s\n", test); return false; }
    char magic[4]; int32_t hdr[7];
    f.read(magic, 4); f.read((char*) hdr, sizeof hdr);
    const int t = hdr[0], k = hdr[1], g = hdr[2], e = hdr[3], h = hdr[4], ell = hdr[5];
    (void) T; (void) K; (void) E; (void) H;
    f.seekg((std::streamoff) (t * h + t * k + t * k + e) * 4, std::ios::cur);
    std::vector<uint8_t> dq(d.d_bytes()), uq(d.u_bytes());
    f.read((char*) dq.data(), (std::streamsize) dq.size());
    f.read((char*) uq.data(), (std::streamsize) uq.size());
    const bool ok = d.d(layer) && d.u(layer) && std::memcmp(d.d(layer), dq.data(), dq.size()) == 0 &&
                    std::memcmp(d.u(layer), uq.data(), uq.size()) == 0;
    std::printf("layer%d D/U byte-equal: %s (t=%d k=%d g=%d h=%d ell=%d)\n", layer, ok ? "yes" : "NO", t, k, g, h, ell);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s dense.bin test_layer0.bin test_layer47.bin\n", argv[0]); return 2; }
    strata::core::LatentDense d;
    std::string err;
    if (!d.open(argv[1], 48, 8, 2560, 1792, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("mapped=%d bytes=%llu d_bytes=%zu u_bytes=%zu\n", (int) d.mapped(), (unsigned long long) d.bytes(),
                d.d_bytes(), d.u_bytes());
    const bool ok = cmp_layer(d, argv[2], 0, 8, 10, 512, 2560) && cmp_layer(d, argv[3], 47, 8, 10, 512, 2560);
    std::printf("RESULT: %s\n", ok ? "ok" : "BAD");
    return ok ? 0 : 1;
}
