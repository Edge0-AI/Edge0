// include/strata/kernels/cpu/hadamard.hpp - the activation-side transform of Hadamard-folded experts.
//
// Some GGUFs (Agention's APR "Gyro" files) store their routed experts in a block-Hadamard-rotated basis: the
// encoder replaced each expert weight W by W' = W * diag(s) * H_b^T per block of b input columns, where H_b is
// the normalized Sylvester-Walsh-Hadamard matrix (entry (-1)^popcount(r & c) / sqrt(b)) and s a +/-1 sign
// vector of the input width.  The GGUF says so in its `prism.hadamard.*` metadata.  Since H_b is orthonormal and
// symmetric, W x = W' (H_b (s * x)): the runtime must apply y = H_b (s * x), per block of b values, to every
// activation that enters a folded weight - the gate/up input x and the down input h - before quantizing it.
// Skipping it on any path (CPU or GPU, decode or prompt) gives plausible-looking garbage, so a pack that carries
// rotated experts is refused by an engine path that cannot apply it.
//
// tools/iq_pack.py reads the metadata and writes `<pack>/hadamard.txt` (and native_experts.txt v5, so an engine
// older than this refuses the pack).  The format:
//
//   # strata hadamard v1 ...                      (comment lines start with #)
//   block 128                                     the rotation block (a power of two)
//   signs <width> <width x +1/-1>                 one line per sign-vector width (none = identity signs)
//   layer <l> <gate 0|1> <up 0|1> <down 0|1>      which roles of layer l are folded (gate == up: one input)
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

struct HadamardSpec {
    int block = 0;                                  ///< 0: no Hadamard-folded experts
    std::map<int, std::vector<float>> signs;        ///< width -> +1/-1 per input value (empty: identity signs)
    std::vector<uint8_t> roles;                     ///< per layer: bit 0 gate, bit 1 up, bit 2 down
    bool any() const { return block > 0; }
    bool gu(int64_t layer) const { return block > 0 && (roles[(size_t) layer] & 3u) != 0; }
    bool down(int64_t layer) const { return block > 0 && (roles[(size_t) layer] & 4u) != 0; }
};

/// Parses `path` (hadamard.txt) for a model of `n_layers` layers.  False with a reason when it is malformed.
bool hadamard_parse(const std::string& path, int64_t n_layers, HadamardSpec& h, std::string& err);

/// The process-wide spec (set by expert_layout_load from the pack; empty for every other pack).
const HadamardSpec& hadamard_spec();
void hadamard_set(HadamardSpec h);
/// The sign vector of `width` in the process-wide spec, or nullptr (identity signs).  The pointer stays valid
/// until the next hadamard_set.
const float* hadamard_signs(int width);

/// In place: x[i] *= signs[i] (when signs is not null), then each block of `block` values -> H_block x / sqrt(block).
/// `n` must be a multiple of `block`.
void hadamard_rotate(float* x, int n, int block, const float* signs);

}  // namespace strata::kernels::cpu
