// metal/tests/ref_dump.cpp - writes random blocks of every type ref_quant knows and their CPU-reference dequant, so
// that the reference itself can be checked against gguf-py (metal/tools/check_ref.py):
//   ref_dump <dir>   ->  <dir>/<TYPE>.bin (raw blocks, 16 rows x 2560) and <dir>/<TYPE>.f32
#include "ref_quant.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const int rows = 16, cols = 2560;
    for (int t : {ref::Q2_0, ref::Q4_0, ref::Q5_0, ref::Q8_0, ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K, ref::IQ2_XXS,
                  ref::IQ2_XS, ref::IQ2_S, ref::IQ3_XXS, ref::IQ3_S, ref::IQ4_NL, ref::IQ4_XS, ref::BF16, ref::F16}) {
        const size_t rb = ref::row_bytes(t, cols);
        std::vector<uint8_t> raw(rb * rows);
        ref::random_blocks(t, raw.data(), rows, cols, 77 + t);
        std::vector<float> f((size_t)rows * cols);
        ref::dequant(t, raw.data(), f.data(), f.size());
        const std::string n = dir + "/" + ref::type_name(t);
        FILE* a = std::fopen((n + ".bin").c_str(), "wb");
        FILE* b = std::fopen((n + ".f32").c_str(), "wb");
        std::fwrite(raw.data(), 1, raw.size(), a);
        std::fwrite(f.data(), 4, f.size(), b);
        std::fclose(a);
        std::fclose(b);
    }
    std::printf("wrote %s\n", dir.c_str());
    return 0;
}
