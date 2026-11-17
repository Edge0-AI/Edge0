// Q2_0 dequant must preserve signed zero when a negative scale meets code 1.
// This caught HIP 5.7/gfx1012 folding the half product to positive zero.
#include "strata/kernels/iq_kernels.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); } } while (0)

int main() {
    // GGUF Q2_0: one half scale followed by sixteen packed 2-bit bytes.
    struct Block { uint16_t d; uint8_t qs[16]; };
    static_assert(sizeof(Block) == 18);
    constexpr int blocks = 16, values = blocks * 64;
    std::vector<Block> input(blocks);
    std::vector<uint16_t> expected(values), got(values);
    for (int b = 0; b < blocks; ++b) {
        const float scale = (b & 1) ? -0.5f : 0.5f;
        const __half half_scale = __float2half(scale);
        std::memcpy(&input[b].d, &half_scale, 2);
        for (int i = 0; i < 16; ++i) input[b].qs[i] = 0xe4;  // codes 0,1,2,3
        for (int i = 0; i < 64; ++i) {
            const __half value = __float2half(scale * float((i % 4) - 1));
            std::memcpy(&expected[b * 64 + i], &value, 2);
        }
    }
    void* weights = nullptr;
    uint16_t* output = nullptr;
    CHECK(hipMalloc(&weights, input.size() * sizeof(Block)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&output), values * sizeof(uint16_t)));
    CHECK(hipMemcpy(weights, input.data(), input.size() * sizeof(Block), hipMemcpyHostToDevice));
    strata::kernels::iq_dequant_f16(42, weights, values, output, nullptr);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(got.data(), output, values * sizeof(uint16_t), hipMemcpyDeviceToHost));
    int failures = 0;
    for (int i = 0; i < values; ++i) {
        if (got[i] == expected[i]) continue;
        if (failures++ < 4)
            std::printf("index=%d got=0x%04x expected=0x%04x\n", i, unsigned(got[i]), unsigned(expected[i]));
    }
    CHECK(hipFree(weights));
    CHECK(hipFree(output));
    std::printf("hip_q2_zero: %d values, %d failures\n", values, failures);
    return failures ? 1 : 0;
}
