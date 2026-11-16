// metal/include/strata/gpu/metal.hpp - what the Metal kernel library needs beyond include/strata/gpu/runtime.hpp.
//
// The host code never includes this: it sees only runtime.hpp and the strata::kernels::* headers.  The Metal
// implementations of those kernel headers (metal/src/kernels/*) turn their `void* stream` into a gpu::Stream and call
// dispatch(): encoded now, or recorded when the stream is being captured (the Metal "graph").
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "strata/gpu/runtime.hpp"

namespace MTL {
class Device;
class ComputePipelineState;
}  // namespace MTL

namespace strata::gpu::metal {

MTL::Device* device();

/// Compile MSL source at run time (the Command Line Tools have no offline `metal` compiler) and return the pipeline
/// of `function`.  Cached by (source name, function name); nullptr and `err` on failure.
MTL::ComputePipelineState* pipeline(const char* source_name, const char* source, const char* function, std::string& err);

struct Grid {
    uint32_t x = 1, y = 1, z = 1;
};

/// One compute dispatch: `args` (<= 4 KiB, copied now, as a CUDA kernel's arguments are at launch/capture) is bound at
/// buffer index 0; the kernel reads its device pointers (gpuAddress values) from it.  `threadgroups` x `threads`
/// like a CUDA <<<grid, block, shared>>>.  Dispatches on one stream are ordered (each one sees the previous one's
/// writes), as kernels on one CUDA stream are.
Error dispatch(Stream s, MTL::ComputePipelineState* p, const void* args, size_t arg_bytes, Grid threadgroups, Grid threads,
               size_t threadgroup_bytes = 0);

/// Counters for tests and the dispatch-cost budget.
struct Stats {
    uint64_t dispatches = 0, command_buffers = 0, blits = 0, host_syncs = 0;
};
Stats stats();

}  // namespace strata::gpu::metal
