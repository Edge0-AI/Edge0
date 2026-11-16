// metal/src/engine/mmvq_host.cpp - the host-side helpers of src/kernels/cuda/native_mmvq.cu the shared host code reads
// (which GGUF types the dense projections take, their byte sizes).  Copied from native_mmvq.cu; keep in step.  The
// kernels are not ported: on Metal the decode window runs mac-kern's GEMVs (metal/src/engine/window.cpp).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int Q8K = 32;
constexpr std::size_t kQ81Block = 4 + 32;   // half2 ds + int8 qs[32]
void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (ncols < 1 || ncols > 8) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
}  // namespace

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
        case 2: block_elems = 32; block_bytes = 18; break;
        case 6: block_elems = 32; block_bytes = 22; break;
        case 7: block_elems = 32; block_bytes = 24; break;
        case 8: block_elems = 32; block_bytes = 34; break;
        case 20: block_elems = 32; block_bytes = 18; break;
        case 11: block_elems = 256; block_bytes = 110; break;
        case 12: block_elems = 256; block_bytes = 144; break;
        case 13: block_elems = 256; block_bytes = 176; break;
        case 14: block_elems = 256; block_bytes = 210; break;
        case 23: block_elems = 256; block_bytes = 136; break;
        case 42: block_elems = 64; block_bytes = 18; break;
        case 16: case 17: case 18: case 21: case 22: case 29:
            block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
        default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out))
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    return row_bytes * std::size_t(n_out);
}

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * kQ81Block;
}
std::size_t native_q8_1_il_bytes(int n_in, int ncols) { return native_q8_1_bytes(n_in, ncols) * 2; }
bool native_mmvq_il_supported(int, int, int) { return false; }
bool native_q8_0_packed_eligible(int, int) { return false; }
bool native_q8_0_packed_enabled() { return false; }
bool native_q6_k_packed_enabled() { return false; }

}  // namespace strata::kernels
