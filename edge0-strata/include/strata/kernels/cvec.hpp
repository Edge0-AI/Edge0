// include/strata/kernels/cvec.hpp - a control vector on the residual stream: the engine's side of llama.cpp's
// `llama_adapter_cvec` with the `--cvec-mode` / `--cvec-dir` extension of the `experimental-speed-projection`
// package (its `02-cvec-projection-mode.patch`, and `01-qwen4exp-cvec-hooks.patch` for WHERE it applies).
//
// After layer l's FFN write - llama.cpp's `build_cvec(res_hc, il)` right after the layer's second
// `build_hc_combine` - every one of the hc residual streams h of every token becomes
//
//   project:  h <- h - s_l (h . v_l) v_l        v_l = d / |d|, s_l = |d|, d = the scaled direction
//   add:      h <- h + d                        llama.cpp's stock additive control vector
//
// for l in [first, last].  `d` is the GGUF's `direction.<src>` times the file's scale (summed over files), with
// src = l (per-layer) or one fixed layer (single:L, project mode only), exactly as llama.cpp's loader places it;
// layer 0 has no direction.  Off unless a vector is loaded (--control-vector-scaled); a loaded one is switched
// per request through a device flag, so the captured graphs are the same either way.  With the flag off the
// result is bit-identical to the engine without a vector: the kernel then only applies the pending residual
// write, with the fused read's own arithmetic.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels {

struct Cvec {
    const float* dir = nullptr;   ///< device, n_layers * n_embd: project = the unit v_l, add = d_l; zero rows elsewhere
    const float* s = nullptr;     ///< device, n_layers: project = s_l, add = 1; 0 = the layer is not steered
    const int* on = nullptr;      ///< device int: 0 = this request runs the stock model
    int mode = 0;                 ///< 0 = project, 1 = add
    int first = 0, last = -1;     ///< the steered layers, inclusive
    int64_t n_embd = 0, hc = 0;
    bool loaded() const { return dir != nullptr; }
    /// Layer l's FFN write is followed by the vector (with a real direction there).  Decided at load time, never
    /// per request: it changes where the residual write happens, so the graphs depend on it.
    bool covers(int64_t l) const { return loaded() && l >= first && l <= last && l < (int64_t) steered.size() && steered[(size_t) l]; }
    std::vector<bool> steered;    ///< host copy of s_l != 0
};

/// The loaded vector (empty until `cvec_upload`).
const Cvec& cvec();

/// Upload a vector built by the loader: `dir` is n_layers * n_embd, `s` n_layers (see `Cvec`).  Starts ON.
bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err);

/// A layer split: put the loaded vector's tables on the CURRENT device too (cvec_apply uses the tables of the device
/// it runs on).  No-op without a vector or when this device has them already.
bool cvec_replicate(std::string& err);

/// The per-request switch, on every device that holds the vector.  Synchronizes them when it changes, so call it
/// between requests.
void cvec_set_enabled(bool on);
bool cvec_enabled();

/// The CURRENT device's tables (dir n_layers x n_embd, s n_layers, the request flag), for a caller that applies the
/// vector inside its own kernel (the prompt path's STRATA_CVEC_FUSE).  False without a vector on this device.
bool cvec_tables(const float** dir, const float** s, const int** on);

/// Layer `layer`'s vector on T tokens' residual stacks (`R + t * r_ld`, hc streams of n_embd).  With `write`, the
/// pending FFN write `R += bo * 2 sigmoid(inj / hc)` (the fused read's arithmetic) is applied first, for callers
/// whose writes are folded into the next layer's read; without it, R must already hold the layer's output.
void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream);

}  // namespace strata::kernels
