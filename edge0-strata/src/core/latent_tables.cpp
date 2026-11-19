// src/core/latent_tables.cpp - see include/strata/core/latent_tables.hpp.
#include "strata/core/latent_tables.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {
// Switches the calling thread to `dev` and back.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};
bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("latent tables: ") + what + ": " + cudaGetErrorString(e);
    return false;
}
}  // namespace

LatentTables::~LatentTables() { close(); }

void LatentTables::bind(LatentDense* dense, const int32_t* host_labels, int64_t n_layers, int64_t E,
                        int64_t n_embd, int64_t latent) {
    close();
    dense_ = dense;
    host_labels_ = host_labels;
    n_layers_ = n_layers;
    E_ = E;
    n_embd_ = n_embd;
    latent_ = latent;
}

void LatentTables::close() {
    for (auto& [device, d] : devs_) {
        On on(device);
        if (d.labels) cudaFree(d.labels);
        if (d.u) cudaFree(d.u);
    }
    devs_.clear();
}

LatentTables::Dev* LatentTables::dev(int device) {
    auto it = devs_.find(device);
    if (it != devs_.end()) return &it->second;
    devs_.emplace(device, Dev{});
    return &devs_[device];
}

bool LatentTables::ensure(int device, int64_t layer, std::string& err) {
    if (dense_ == nullptr || host_labels_ == nullptr) { err = "latent tables: not bound"; return false; }
    if (layer < 0 || layer >= n_layers_) { err = "latent tables: layer out of range"; return false; }
    Dev* d = dev(device);
    On on(device);

    // labels: the whole [n_layers * E] table, tiny (~2 KiB per layer), uploaded once per device.
    if (d->labels == nullptr) {
        const size_t bytes = (size_t) n_layers_ * (size_t) E_ * sizeof(int32_t);
        if (!ck(cudaMalloc((void**) &d->labels, bytes), "labels alloc", err)) return false;
        if (!ck(cudaMemcpy(d->labels, host_labels_, bytes, cudaMemcpyHostToDevice), "labels upload", err)) return false;
    }

    // U: one layer resident at a time (decode serves one layer per dispatch; a later ensure() re-ups).
    const size_t ubytes = dense_->u_bytes();
    if (d->u == nullptr) {
        if (!ck(cudaMalloc((void**) &d->u, ubytes), "U alloc", err)) return false;
        d->u_layer = -1;
    }
    if (d->u_layer != layer) {
        if (!dense_->mapped()) { err = "latent tables: the dense file is not open"; return false; }
        const uint8_t* host = dense_->u(layer);
        if (!ck(cudaMemcpy(d->u, host, ubytes, cudaMemcpyHostToDevice), "U upload", err)) return false;
        d->u_layer = layer;
    }
    return true;
}

const uint8_t* LatentTables::u_dev(int device, int64_t layer) const {
    auto it = devs_.find(device);
    if (it == devs_.end()) return nullptr;
    return it->second.u_layer == layer ? it->second.u : nullptr;
}

const int32_t* LatentTables::labels_dev(int device) const {
    auto it = devs_.find(device);
    return it == devs_.end() ? nullptr : it->second.labels;
}

}  // namespace strata::core