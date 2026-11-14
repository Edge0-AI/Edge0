// metal/include/metal_compat/cpuid.h - GCC's <cpuid.h> on arm64: every leaf reads as zero, so the x86 feature
// probes (src/kernels/cpu/expert_layout.cpp) report no AVX / AVX-512 / VNNI and the engine never picks an x86 kernel.
#pragma once
#define __cpuid(leaf, a, b, c, d) ((void) (leaf), (a) = 0, (b) = 0, (c) = 0, (d) = 0)
#define __cpuid_count(leaf, sub, a, b, c, d) ((void) (leaf), (void) (sub), (a) = 0, (b) = 0, (c) = 0, (d) = 0)
static inline unsigned __get_cpuid_max(unsigned, unsigned* sig) { if (sig) *sig = 0; return 0; }
