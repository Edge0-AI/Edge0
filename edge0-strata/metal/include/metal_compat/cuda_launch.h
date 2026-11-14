// cuda_launch.h - the <<<grid, block, shared, stream>>> launch syntax, as a call.
//
// A C++ compiler cannot parse CUDA's launch syntax, and the Metal port replaces every launch with a dispatch anyway (the grid
// becomes a threadgroup count, the block a threadgroup size).  strata_ds4_launch() is that call: the kernel first, then the
// launch shape, then the kernel's own arguments - every parameter generic, because the shapes vary (dim3, plain ints, sizes,
// and the stream type), and the point of this header is to make the sources parse, not to type-check the port.
//
// Until a kernel has a Metal implementation the call stops the run naming the kernel, so the engine reports the first one it
// needs and the port proceeds one kernel at a time.
#pragma once

#include <cstdio>
#include <cstdlib>

template <typename K, typename... A>
static inline void strata_ds4_launch(K kernel, A&&...) {
    static_assert(sizeof(K) > 0, "kernel");
    std::fprintf(stderr, "strata metal: a dsv41 kernel launch has no Metal dispatch yet\n");
    std::abort();
}

// the call sites sit inside namespaces whose first argument would win argument-dependent lookup: force the global one
// the call sites carry their own location, so a run names the exact launch that needs a dispatch
#define strata_ds4_launch(...) ::strata_ds4_launch_at(__FILE__, __LINE__, __VA_ARGS__)

// the macro adds the call site's location, so a run names the exact launch that still needs a Metal dispatch
template <typename K, typename... A>
static inline void strata_ds4_launch_at(const char* file, int line, K, A&&...) {
    std::fprintf(stderr, "strata metal: kernel launch with no Metal dispatch at %s:%d\n", file, line);
    std::abort();
}
