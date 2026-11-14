// metal/src/engine/stubs_ds4.cpp - CPU-kernel entry points the DeepSeek engine (strata_ds4) references that have no Apple
// implementation.  strata::ds4::gguf::dots() asks iq256x_supported() before it uses iq256x_rows(); on arm64 the answer is no and the
// CPU lane keeps ggml-cpu's rows, which have their own NEON kernels.
#include "strata/kernels/cpu/iq_avx2.hpp"

#include <cstdlib>

namespace strata::kernels::cpu {

bool iq256x_supported(int) noexcept { return false; }

void iq256x_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    std::abort();   // unreachable: iq256x_supported() is false here, and callers check it
}

}   // namespace strata::kernels::cpu
