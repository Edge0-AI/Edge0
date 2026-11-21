#pragma once

#include "strata/core/latent_dense.hpp"
#include <cuda_runtime.h>

namespace strata::core {

// The caller owns the result on the current device. The mapped file and labels keep global layer indices.
inline bool upload_latent_du(const LatentDense& dense, int64_t first, int64_t end, LatentDeviceDU& out, std::string& err) {
    LatentDeviceDU slice{nullptr, nullptr, first};
    if (first < 0 || end <= first || end > dense.layers()) {
        err = "invalid latent D/U layer range";
        return false;
    }
    const size_t count = static_cast<size_t>(end - first);
    auto rc = cudaMalloc(reinterpret_cast<void**>(&slice.d), count * dense.d_bytes());
    if (rc == cudaSuccess) rc = cudaMalloc(reinterpret_cast<void**>(&slice.u), count * dense.u_bytes());
    for (int64_t l = first; rc == cudaSuccess && l < end; ++l) {
        rc = cudaMemcpy(slice.d_at(l, dense), dense.d(l), dense.d_bytes(), cudaMemcpyHostToDevice);
        if (rc == cudaSuccess) rc = cudaMemcpy(slice.u_at(l, dense), dense.u(l), dense.u_bytes(), cudaMemcpyHostToDevice);
    }
    if (rc != cudaSuccess) {
        err = "latent D/U upload for layers [" + std::to_string(first) + ", " + std::to_string(end) + "): " + cudaGetErrorString(rc);
        if (slice.d) cudaFree(slice.d);
        if (slice.u) cudaFree(slice.u);
        return false;
    }
    out = slice;
    return true;
}

}  // namespace strata::core
