// metal/tests/coop_layout.cpp - which (index 0, index 1) elements of a single-simdgroup matmul2d's right input, left
// input and destination cooperative tensors each lane holds (mm_bench.metal coop_layout).
//   coop_layout [TM_TN_TK]
#include "mtl_rt.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    const std::string cfg = argc > 1 ? argv[1] : "16_32_16";
    mtl::Context cx;
    const int lib = cx.compile({"mm_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Pipeline p = cx.pipeline(lib, "coop_layout_" + cfg);
    mtl::Buffer out = cx.alloc(4 * 3 * 32 * 128 * 4);
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(out)};
    d.tg[0] = 32;
    cx.run({d});
    const int* o = out.as<int>();
    const char* nm[3] = {"right", "left", "dest"};
    for (int w = 0; w < 3; ++w) {
        std::printf("%s (tile %s):\n", nm[w], cfg.c_str());
        for (int l = 0; l < 32; ++l) {
            const int* q = o + (w * 32 + l) * 128 * 4;
            std::printf(" l%-2d cap %d:", l, q[2]);
            for (int i = 0; i < q[2] && i < 128; ++i) std::printf(" (%d,%d)", q[i * 4], q[i * 4 + 1]);
            std::printf("\n");
        }
    }
    return 0;
}
