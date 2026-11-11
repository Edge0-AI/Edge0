// Numerical parity for the CUDA intrinsics supplied by the HIP compatibility layer.
#include <hip/hip_runtime.h>
#include "strata/hip_compat/intrinsics.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>

#define CHECK(call)                                                                                                  \
    do {                                                                                                             \
        const hipError_t error = (call);                                                                             \
        if (error != hipSuccess) {                                                                                   \
            std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error));                                     \
            return 2;                                                                                                \
        }                                                                                                            \
    } while (0)

namespace {

constexpr int kLanes = 32;
constexpr int kIntegerResults = 10;

__global__ void intrinsic_probe(int* integer_results, float* float_shuffle, double* double_shuffle,
                                unsigned* ballot, const uint32_t* dot_inputs) {
    const int lane = static_cast<int>(threadIdx.x);
    constexpr unsigned mask = 0xffffffffu;
    const uint32_t a = 0x80ff017fu;
    const uint32_t b = 0x0203fe81u;

    integer_results[lane * kIntegerResults + 0] = __dp4a(static_cast<int>(a), static_cast<int>(b), 7);
    integer_results[lane * kIntegerResults + 1] = __dp4a(0x7f7f7f7f, 0x7f7f7f7f, 0x7fffffff);
    integer_results[lane * kIntegerResults + 2] = __vsub4(0x00ff0102, 0x01010103);
    integer_results[lane * kIntegerResults + 3] = __vsub4(0x00000000, 0x01020304);
    integer_results[lane * kIntegerResults + 4] = __vsubss4(0x807f0080, 0x017f0180);
    integer_results[lane * kIntegerResults + 5] = __vcmpne4(0x00ff0102, 0x01010103);
    integer_results[lane * kIntegerResults + 6] =
        __dp4a(static_cast<int>(dot_inputs[lane * 2]), static_cast<int>(dot_inputs[lane * 2 + 1]),
               0x7fffffff - lane);

    __shared__ int exchanged[kLanes];
    int sum = 0;
    for (int iteration = 0; iteration < 64; ++iteration) {
        exchanged[lane] = static_cast<int>(dot_inputs[lane * 2] & 255u) + iteration;
        __syncwarp();
        sum += exchanged[lane ^ 1];
        __syncwarp();
    }
    integer_results[lane * kIntegerResults + 7] = sum;
    const int input_a = static_cast<int>(dot_inputs[lane * 2]);
    const int input_b = static_cast<int>(dot_inputs[lane * 2 + 1]);
    integer_results[lane * kIntegerResults + 8] = __dp4a(input_a, input_b, input_a);
    integer_results[lane * kIntegerResults + 9] = __dp4a(input_a, input_b, input_b);

    float_shuffle[lane * 4 + 0] = __shfl_xor_sync(mask, static_cast<float>(lane), 1);
    float_shuffle[lane * 4 + 1] = static_cast<float>(__shfl_down_sync(mask, lane, 4, 8));
    float_shuffle[lane * 4 + 2] = static_cast<float>(__shfl_up_sync(mask, lane, 1));
    float_shuffle[lane * 4 + 3] = static_cast<float>(__shfl_sync(mask, lane, 0));
    double_shuffle[lane] = __shfl_xor_sync(mask, static_cast<double>(lane) + 0.25, 1);

    const unsigned hit = __ballot_sync(mask, (lane & 1) == 0);
    if (lane == 0) {
        *ballot = hit;
        __nanosleep(100);
    }
}

// The byte-lane intrinsics over many operands: __byte_perm with selector bits it ignores (bit 3 of each nibble and
// the upper half), since the i-quant table lookups pass data-dependent selectors; the packed subtracts and compare
// on every sign and overflow combination (the compare's second operand keeps two lanes of the first).
constexpr int kPermCases = 1024;
constexpr int kPermResults = 4;
__global__ void byte_perm_probe(const uint32_t* inputs, uint32_t* results) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= kPermCases) return;
    const uint32_t x = inputs[3 * i], y = inputs[3 * i + 1], s = inputs[3 * i + 2];
    results[kPermResults * i + 0] = __byte_perm(x, y, s);
    results[kPermResults * i + 1] = static_cast<uint32_t>(__vsub4(static_cast<int>(x), static_cast<int>(y)));
    results[kPermResults * i + 2] = static_cast<uint32_t>(__vsubss4(static_cast<int>(x), static_cast<int>(y)));
    results[kPermResults * i + 3] =
        static_cast<uint32_t>(__vcmpne4(static_cast<int>(x), static_cast<int>(x ^ (s & 0xff00ff00u))));
}

uint32_t byte_perm_reference(uint32_t x, uint32_t y, uint32_t s) {
    const uint64_t pair = (static_cast<uint64_t>(y) << 32) | x;
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) out |= static_cast<uint32_t>((pair >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
    return out;
}

int signed_byte(uint32_t word, int lane) {
    const unsigned value = (word >> (lane * 8)) & 0xffu;
    return value < 0x80u ? static_cast<int>(value) : static_cast<int>(value) - 0x100;
}

int dp4a_reference(uint32_t a, uint32_t b, int c) {
    int64_t sum = c;
    for (int lane = 0; lane < 4; ++lane) sum += signed_byte(a, lane) * signed_byte(b, lane);
    return static_cast<int32_t>(static_cast<uint32_t>(sum));
}

uint32_t byte_sub_reference(uint32_t a, uint32_t b) {
    uint32_t out = 0;
    for (int lane = 0; lane < 4; ++lane)
        out |= ((((a >> (lane * 8)) & 0xffu) - ((b >> (lane * 8)) & 0xffu)) & 0xffu) << (lane * 8);
    return out;
}

uint32_t signed_saturating_sub_reference(uint32_t a, uint32_t b) {
    uint32_t out = 0;
    for (int lane = 0; lane < 4; ++lane) {
        int value = signed_byte(a, lane) - signed_byte(b, lane);
        if (value < -128) value = -128;
        if (value > 127) value = 127;
        out |= (static_cast<uint32_t>(value) & 0xffu) << (lane * 8);
    }
    return out;
}

uint32_t not_equal_reference(uint32_t a, uint32_t b) {
    uint32_t out = 0;
    for (int lane = 0; lane < 4; ++lane)
        if (((a >> (lane * 8)) & 0xffu) != ((b >> (lane * 8)) & 0xffu)) out |= 0xffu << (lane * 8);
    return out;
}

}  // namespace

int main() {
    int device = 0;
    CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    CHECK(hipGetDeviceProperties(&properties, device));
    if (properties.warpSize != kLanes) {
        std::fprintf(stderr, "expected wave32, got wave%d\n", properties.warpSize);
        return 1;
    }

    int* device_integers = nullptr;
    float* device_float_shuffle = nullptr;
    double* device_double_shuffle = nullptr;
    unsigned* device_ballot = nullptr;
    uint32_t* device_dot_inputs = nullptr;
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_integers), kLanes * kIntegerResults * sizeof(int)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_float_shuffle), kLanes * 4 * sizeof(float)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_double_shuffle), kLanes * sizeof(double)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_ballot), sizeof(unsigned)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_dot_inputs), kLanes * 2 * sizeof(uint32_t)));

    uint32_t dot_inputs[kLanes * 2]{};
    for (int lane = 0; lane < kLanes; ++lane) {
        dot_inputs[lane * 2] = 0x80ff017fu ^ (0x01010101u * static_cast<uint32_t>(lane));
        dot_inputs[lane * 2 + 1] = 0x0203fe81u + (0x11111111u * static_cast<uint32_t>(lane));
    }
    CHECK(hipMemcpy(device_dot_inputs, dot_inputs, sizeof(dot_inputs), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(intrinsic_probe, dim3(1), dim3(kLanes), 0, 0,
                       device_integers, device_float_shuffle, device_double_shuffle, device_ballot, device_dot_inputs);
    CHECK(hipDeviceSynchronize());

    int integers[kLanes * kIntegerResults]{};
    float float_shuffle[kLanes * 4]{};
    double double_shuffle[kLanes]{};
    unsigned ballot = 0;
    CHECK(hipMemcpy(integers, device_integers, sizeof(integers), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(float_shuffle, device_float_shuffle, sizeof(float_shuffle), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(double_shuffle, device_double_shuffle, sizeof(double_shuffle), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&ballot, device_ballot, sizeof(ballot), hipMemcpyDeviceToHost));

    constexpr uint32_t a = 0x80ff017fu, b = 0x0203fe81u;
    const uint32_t expected_integer[6] = {
        static_cast<uint32_t>(dp4a_reference(a, b, 7)),
        static_cast<uint32_t>(dp4a_reference(0x7f7f7f7fu, 0x7f7f7f7fu, std::numeric_limits<int>::max())),
        byte_sub_reference(0x00ff0102u, 0x01010103u),
        byte_sub_reference(0x00000000u, 0x01020304u),
        signed_saturating_sub_reference(0x807f0080u, 0x017f0180u),
        not_equal_reference(0x00ff0102u, 0x01010103u),
    };
    bool ok = ballot == 0x55555555u;
    for (int lane = 0; lane < kLanes; ++lane) {
        for (int i = 0; i < 6; ++i)
            ok = ok && static_cast<uint32_t>(integers[lane * kIntegerResults + i]) == expected_integer[i];
        ok = ok && static_cast<uint32_t>(integers[lane * kIntegerResults + 6]) ==
                     static_cast<uint32_t>(dp4a_reference(dot_inputs[lane * 2], dot_inputs[lane * 2 + 1],
                                                          0x7fffffff - lane));
        ok = ok && integers[lane * kIntegerResults + 7] ==
                     64 * static_cast<int>(dot_inputs[(lane ^ 1) * 2] & 255u) + 2016;
        for (int alias = 0; alias < 2; ++alias)
            ok = ok && static_cast<uint32_t>(integers[lane * kIntegerResults + 8 + alias]) ==
                static_cast<uint32_t>(dp4a_reference(dot_inputs[lane * 2], dot_inputs[lane * 2 + 1],
                    static_cast<int>(dot_inputs[lane * 2 + alias])));
        ok = ok && float_shuffle[lane * 4 + 0] == static_cast<float>(lane ^ 1);
        const int down = lane % 8 < 4 ? lane + 4 : lane;
        ok = ok && float_shuffle[lane * 4 + 1] == static_cast<float>(down);
        ok = ok && float_shuffle[lane * 4 + 2] == static_cast<float>(lane == 0 ? 0 : lane - 1);
        ok = ok && float_shuffle[lane * 4 + 3] == 0.0f;
        ok = ok && double_shuffle[lane] == static_cast<double>(lane ^ 1) + 0.25;
    }

    uint32_t perm_inputs[kPermCases * 3]{};
    uint32_t state = 0x9e3779b9u;
    for (uint32_t& v : perm_inputs) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        v = state;
    }
    uint32_t* device_perm_inputs = nullptr;
    uint32_t* device_perm_results = nullptr;
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_perm_inputs), sizeof(perm_inputs)));
    CHECK(hipMalloc(reinterpret_cast<void**>(&device_perm_results), kPermCases * kPermResults * sizeof(uint32_t)));
    CHECK(hipMemcpy(device_perm_inputs, perm_inputs, sizeof(perm_inputs), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(byte_perm_probe, dim3(kPermCases / 256), dim3(256), 0, 0, device_perm_inputs, device_perm_results);
    CHECK(hipDeviceSynchronize());
    uint32_t perm_results[kPermCases * kPermResults]{};
    CHECK(hipMemcpy(perm_results, device_perm_results, sizeof(perm_results), hipMemcpyDeviceToHost));
    for (int i = 0; i < kPermCases; ++i) {
        const uint32_t x = perm_inputs[3 * i], y = perm_inputs[3 * i + 1], s = perm_inputs[3 * i + 2];
        const uint32_t* r = perm_results + kPermResults * i;
        ok = ok && r[0] == byte_perm_reference(x, y, s);
        ok = ok && r[1] == byte_sub_reference(x, y);
        ok = ok && r[2] == signed_saturating_sub_reference(x, y);
        ok = ok && r[3] == not_equal_reference(x, x ^ (s & 0xff00ff00u));
    }
    CHECK(hipFree(device_perm_results));
    CHECK(hipFree(device_perm_inputs));

    CHECK(hipFree(device_ballot));
    CHECK(hipFree(device_dot_inputs));
    CHECK(hipFree(device_double_shuffle));
    CHECK(hipFree(device_float_shuffle));
    CHECK(hipFree(device_integers));
    if (!ok) {
        std::fprintf(stderr, "HIP intrinsic parity failed (ballot 0x%08x)\n", ballot);
        return 1;
    }
    std::puts("HIP intrinsics parity OK: dynamic signed dot4/overflow, byte permute, packed integer boundaries, wave32 shuffles, shared-memory exchange, ballot and sleep");
    return 0;
}
