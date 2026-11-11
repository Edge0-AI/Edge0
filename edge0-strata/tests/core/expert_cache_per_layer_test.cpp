// #369: per-layer admission on a sized cache (open_sized, every native pack) - each layer admits into its own slot
// range, as on the uniform cache, so no two layers write the same slot.  Needs a CUDA device (exits 77 without one).
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    std::string err;
    for (int sized = 0; sized < 2; ++sized) {
        strata::core::ExpertCache cache;
        const bool ok = sized ? cache.open_sized(std::vector<int64_t>(6, 1024), 3, 4, err)
                              : cache.open(6, 3, 4, 1024, err);
        if (!ok) {
            std::fprintf(stderr, "open: %s\n", err.c_str());
            return 1;
        }
        cache.set_per_layer_admission(true);
        for (int64_t layer = 0; layer < 3; ++layer) {
            int64_t lo = 0, hi = 0;
            cache.layer_slot_range(layer, lo, hi);
            for (int64_t e = 0; e < 3; ++e) {
                const int32_t slot = cache.admit(layer, e);
                const bool want_room = lo + e < hi;
                if (want_room ? slot != (int32_t) (lo + e) : slot != strata::core::kNotResident) {
                    std::fprintf(stderr, "%s cache: layer %lld expert %lld got slot %d (range %lld..%lld)\n",
                                 sized ? "sized" : "uniform", (long long) layer, (long long) e, slot, (long long) lo,
                                 (long long) hi - 1);
                    return 1;
                }
            }
        }
    }
    std::puts("expert cache per-layer admission: uniform and sized caches passed");
    return 0;
}
