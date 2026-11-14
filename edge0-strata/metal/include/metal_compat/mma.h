// mma.h - the wmma surface on Metal: the names parse, and using one stops the run.
//
// The DeepSeek token path does not use tensor cores.  The prompt path has an MMA variant written for CUDA/HIP warp matrices;
// Metal's equivalent is the simdgroup matrix API, which is a different shape, so that variant is a later piece of the port.
// These aliases exist so the shared sources compile (which is how the rest of the API surface is enumerated), and every
// operation aborts with a message naming this file - an accidental call cannot return plausible garbage.
#pragma once

#include <cstdio>
#include <cstdlib>

namespace nvcuda {
namespace wmma {

// CUDA 12 spells these as tags: fragment<accumulator, 16, 16, 16, float>
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

template <typename Use, int M, int N, int K, typename T, typename Layout = void> struct fragment {
    T x[M * N];
};

[[noreturn]] inline void strata_metal_wmma_missing(const char* what) {
    std::fprintf(stderr, "strata metal: wmma::%s has no Metal implementation (simdgroup matrices are the equivalent)\n", what);
    std::abort();
}

template <typename... A> inline void fill_fragment(A&&...) { strata_metal_wmma_missing("fill_fragment"); }
template <typename... A> inline void load_matrix_sync(A&&...) { strata_metal_wmma_missing("load_matrix_sync"); }
template <typename... A> inline void mma_sync(A&&...) { strata_metal_wmma_missing("mma_sync"); }   // variadic: the engine passes four fragments of different types
template <typename... A> inline void store_matrix_sync(A&&...) { strata_metal_wmma_missing("store_matrix_sync"); }

}   // namespace wmma
}   // namespace nvcuda
