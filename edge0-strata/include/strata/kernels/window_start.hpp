// include/strata/kernels/window_start.hpp - the first cell the draft layer's attention reads.
//
// The draft layer attends to the last `window` cells.  KVA (docs/KVA.md) can leave cells in the draft layer's K/V
// that were built from approximated rows; `floor` is the first cell that holds a real one, so the attention starts
// there when it is higher.  floor = 0 (every session KVA did not approximate) gives exactly the plain window.
// One function for the device kernel and for the host test.
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_WINDOW_START_HD __host__ __device__
#else
#define STRATA_WINDOW_START_HD
#endif

namespace strata::kernels {

/// The first cell of the attention for a sequence of `n_kv` cells.  The width n_kv - start is at least 1 whenever
/// n_kv is (a floor at or past the end must not leave the attention with nothing to read).
STRATA_WINDOW_START_HD inline int window_start(int n_kv, int window, int floor) {
    int start = n_kv > window ? n_kv - window : 0;
    if (floor > start) start = floor;
    if (start > n_kv - 1) start = n_kv > 0 ? n_kv - 1 : 0;
    return start;
}

}  // namespace strata::kernels
