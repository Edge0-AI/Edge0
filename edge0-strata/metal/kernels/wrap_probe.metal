// wrap_probe.metal - the tests/wrap_split_test.cpp kernel: reads a window at the start of a bound range and another at
// its far end (`far` floats on), to prove a chunked no-copy buffer (Context::wrap with a span) binds the whole range.
#include <metal_stdlib>
using namespace metal;

kernel void probe_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                       device const uint* par [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint n = par[0], far = par[1];
    if (i >= n) return;
    dst[i] = src[i];
    dst[n + i] = src[far + i];
}
