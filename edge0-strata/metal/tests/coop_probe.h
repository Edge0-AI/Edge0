// metal/tests/coop_probe.h - which element layout this GPU gives matmul2d's single-simdgroup right-input cooperative
// tensor: true = runs of 4 consecutive k at one n per lane (the M5), false = anything else (the M4: runs of 2 n at one
// k).  The kernels that fill right inputs per lane (mm_gu3, mm_down3, qsa_attn_mm, qsa_flash) take it as function
// constant 7.  COOP_K4=0/1 in the environment overrides.
#pragma once
#include "mtl_rt.h"

#include <cstdlib>

inline bool coop_k4(mtl::Context& cx) {
    if (const char* e = std::getenv("COOP_K4")) return std::atoi(e) != 0;
    const int lib = cx.compile({"mm_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    for (const char* tile : {"coop_layout_16_64_32", "coop_layout_16_32_32"}) {
    mtl::Pipeline p = cx.pipeline(lib, tile);
    mtl::Buffer out = cx.alloc(4 * 3 * 32 * 128 * 4);
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(out)};
    d.tg[0] = 32;
    cx.run({d});
    // every lane: elements 4j .. 4j + 3 are (k, n), (k + 1, n), (k + 2, n), (k + 3, n), and element i + cap / 2 is
    // (k + 16, n) with k % 32 < 16 (mm_gu3's paired-run decode relies on it)
    const int* o = out.as<int>();
    for (int l = 0; l < 32; ++l) {
        const int* q = o + l * 128 * 4;
        const int cap = q[2];
        for (int i = 0; i + 3 < cap; i += 4)
            for (int j = 1; j < 4; ++j)
                if (q[(i + j) * 4] != q[i * 4] + j || q[(i + j) * 4 + 1] != q[i * 4 + 1]) return false;
        for (int i = 0; i < cap / 2; i += 4)
            if (q[(i + cap / 2) * 4] != q[i * 4] + 16 || q[(i + cap / 2) * 4 + 1] != q[i * 4 + 1] || (q[i * 4] & 31) >= 16)
                return false;
    }
    }
    return true;
}
