// metal/tests/dsu_ref_vs_ggml.cpp - ref_quant.cpp's IQ1_S / IQ1_M / Q6_K dequantizers against ggml's own to_float (the pinned llama.cpp
// tree the engine build links), bit for bit.  Built only with the engine (needs ggml-base).
#include "ref_quant.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    int fails = 0;
    for (int ty : {ref::IQ1_S, ref::IQ1_M, ref::Q6_K, ref::IQ2_XXS}) {
        const int n = 64, k = 2048;
        std::vector<uint8_t> w(ref::row_bytes(ty, k) * n);
        ref::random_blocks(ty, w.data(), n, k, 21 + ty, 0.05f);
        std::vector<float> a((size_t)n * k), b((size_t)n * k);
        ref::dequant(ty, w.data(), a.data(), a.size());
        const ggml_type_traits* tr = ggml_get_type_traits((ggml_type)ty);
        if (!tr || !tr->to_float) { std::printf("%s: ggml has no to_float\n", ref::type_name(ty)); return 2; }
        tr->to_float(w.data(), b.data(), (int64_t)b.size());
        const size_t bad = std::memcmp(a.data(), b.data(), a.size() * 4) == 0 ? 0 : 1;
        size_t nbad = 0;
        for (size_t i = 0; i < a.size(); ++i) nbad += std::memcmp(&a[i], &b[i], 4) != 0;
        std::printf("%-8s ref_quant vs ggml to_float: %zu of %zu values differ  %s\n", ref::type_name(ty), nbad, a.size(), nbad == 0 ? "PASS" : "FAIL");
        fails += (bad || nbad);
    }
    return fails ? 1 : 0;
}
