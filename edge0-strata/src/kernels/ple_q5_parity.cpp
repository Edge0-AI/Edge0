// Real-artifact Q5_0 / Q5_1 / Q8_0 PLE rows against ggml's reference dequantizer, through both readers (the mapped
// one and the direct SSD reader).
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Largest absolute difference between the table's rows and ggml's dequantization of the same bytes, single rows
// and one 16-row batch; -1 when the table cannot be opened or read.
double check(const char* path, strata::kernels::PleIo mode, const uint8_t* bytes, size_t row_bytes,
             const ggml_type_traits* traits, std::string& format, size_t& n_rows) {
    strata::kernels::PleIoOptions options;
    options.mode = mode;
    strata::kernels::PleTable table;
    std::string err;
    if (!table.open(path, err, options)) {
        std::fprintf(stderr, "open: %s\n", err.c_str());
        return -1.0;
    }
    format = table.format();
    n_rows = (size_t) table.rows();
    const uint32_t probes[] = {0, 1, 12345, 20000003, (uint32_t) (table.rows() - 1)};
    double max_abs = 0.0;
    for (uint32_t row : probes) {
        float got[160], want[160];
        table.read_row(row, got);
        traits->to_float(bytes + (size_t) row * row_bytes, want, 160);
        for (int i = 0; i < 160; ++i)
            max_abs = std::max(max_abs, (double) std::fabs(got[i] - want[i]));
    }
    uint32_t rows[16];
    for (int i = 0; i < 16; ++i) rows[i] = probes[i % 5];
    float batch[16 * 160];
    if (!table.issue(rows) || !table.collect(batch, err)) {
        std::fprintf(stderr, "collect: %s\n", err.c_str());
        return -1.0;
    }
    for (int h = 0; h < 16; ++h) {
        float want[160];
        traits->to_float(bytes + (size_t) rows[h] * row_bytes, want, 160);
        for (int i = 0; i < 160; ++i)
            max_abs = std::max(max_abs, (double) std::fabs(batch[h * 160 + i] - want[i]));
    }
    return max_abs;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ple_q5_parity <gguf-containing-ple>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const auto* tensor = gguf.find("per_layer_token_embd.weight");
    // Q5_0 (#296: OrcaRouter's Q4_K_S), Q5_1 (a Q5_K_M GGUF), Q8_0 (UD-Q6_K_XL, Swift-1.5 Q4_K_L): 5 blocks of
    // 32 per row
    if (!tensor || (tensor->type != GGML_TYPE_Q5_0 && tensor->type != GGML_TYPE_Q5_1 && tensor->type != GGML_TYPE_Q8_0) ||
        tensor->shape.size() != 2 || tensor->shape[0] != 160) {
        std::fprintf(stderr, "expected a Q5_0, Q5_1 or Q8_0 PLE [160, N]\n");
        return 2;
    }
    const auto type = (ggml_type) tensor->type;
    const size_t row_bytes = type == GGML_TYPE_Q5_0 ? 110 : type == GGML_TYPE_Q5_1 ? 120 : 170;
    const auto* traits = ggml_get_type_traits(type);
    const uint8_t* bytes = gguf.tensor_data(*tensor);
    bool ok = true;
    for (auto mode : {strata::kernels::PleIo::Mmap, strata::kernels::PleIo::Direct}) {
        std::string format;
        size_t n_rows = 0;
        const double max_abs = check(argv[1], mode, bytes, row_bytes, traits, format, n_rows);
        const bool pass = max_abs >= 0.0 && max_abs <= 1e-6;
        std::printf("%s PLE, %s reader: %zu rows, max_abs %.3e %s\n", format.c_str(),
                    mode == strata::kernels::PleIo::Mmap ? "mapped" : "direct", n_rows, max_abs, pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }
    return ok ? 0 : 1;
}
