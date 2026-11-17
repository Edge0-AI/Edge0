// tools/trellis/tqenc.hpp - tail-biting Viterbi ENCODER for the trellis expert formats (TQ2_T, TQK6, TQK7 of the
// agentionai llama.cpp fork, ggml types 144-146) and two experimental lower-rate versions (TQK5X, TQK4X, and the table-based TQK5T, TQK4T).
//
// Standalone (no ggml, no Strata engine): it only needs third_party/ggml/ggml-common-tq.h (the 2048-point codebook).
// The decoders it must match are src/kernels/cpu/tq_states.hpp / tq_cpu.cpp (tools/trellis/tqenc_test.cpp checks
// bit for bit against them, and the scalar / AVX2 / AVX-512 kernels against the encoder's own reconstruction).
//
// THE FORMAT (see tq_states.hpp, include/strata/kernels/cpu/tq_cpu.hpp, docs/GYRO.md):
//   block = 128 weights = {fp16 d (little endian); qs[4*K]}, 32 steps t of 4 weights (weights 4t..4t+3).
//   State of step t: 15 bits.  weights = d * codebook(state); the state's hash x = state * 0x9e3779b1 picks two
//   points lut[x >> 21] and lut[(x >> 10) & 2047] (fp16 pairs) = four values.  The decoder works on the int16 table
//   round(fp16 * 8192) (tq_state_table), so that table, divided by 8192, is the codebook here.
//   TQK (K = 4,5,6,7): state(t) = bits [(31-t)K, (31-t)K + 15) of the qs bit stream (LSB first), circular modulo
//        32K bits.  With u = 31 - t the states form a shift trellis: state(u+1) = (state(u) >> K) | (new K bits <<
//        (15-K)); the K low bits of state(u) are stream symbol u.  Tail-biting: state(0) closes the cycle.
//   TQ2_T (K = 8): state(t) = ((qs[(t+31)%32] << 8) | qs[t]) & 0x7fff: a shift trellis in the other direction
//        (state(t) = ((state(t-1) & 0x7f) << 8) | qs[t]).
// Both are a "de Bruijn" trellis of 2^15 states with 2^K branches per state, so one Viterbi step is a min over G
// groups of predecessors followed by a broadcast add (G = 2^(15-K) groups for TQK, 128 for TQ2_T), not 2^15 * 2^K
// additions.  Tail-biting = a cycle of 32 states: the group of state(0) as successor equals the group of state(31)
// as predecessor.  The search fixes that group c (exact: all G values; approximate: c taken from a free Viterbi pass
// on the cyclically rotated sequence, QTIP's approach, optionally the best of several rotations).
//
// New experimental types (EXPERIMENTAL_VERSION 2; not in any ggml enum or GGUF reader; ids away from the fork's 143-149):
//   TQK5X (224, 22 bytes / 128 weights, 1.375 bpw) and TQK4X (225, 18 bytes, 1.125 bpw): the TQK rule with K = 5 / 4 and the
//     COMPUTED codebook (src/kernels/cpu/tq_computed.hpp: two LCG words per state masked into four fp16, no table);
//   TQK5T (226) and TQK4T (227): version 1, the same K = 5 / 4 layout with the fork's hashed point table (what the first
//     pilots used).  Same bit order, circular stream, 15-bit mask and fp16 scale in all of them.
#pragma once

#include <cstddef>
#include <cstdint>

namespace tqenc {

inline constexpr int kBlock = 128;
inline constexpr int kStates = 1 << 15;
inline constexpr int kExperimentalVersion = 2;

// ggml type ids of the fork (decoders exist) and the experimental ones (encoder + Strata CPU decoder only)
enum Type : int {
    TQ2_T = 144, TQK6 = 145, TQK7 = 146,    // the fork's types (hashed point table)
    TQK5X = 224, TQK4X = 225,               // experimental version 2: K = 5 / 4 with the computed (LUT-free) codebook, tq_computed.hpp
    TQK5T = 226, TQK4T = 227                // experimental version 1: K = 5 / 4 with the fork's point table (the first pilots)
};
/// Whether the type's codebook is the computed one (TQK5X, TQK4X).
bool type_computed(int type);
/// rms of the codebook values (the first scale guess is rms(block) / this): 1.072 for the table, 0.88 for the computed code.
double codebook_rms(int type);

/// bits per step (8, 6, 7, 5, 4), 0 for any other type.
int type_kbits(int type);
inline int block_bytes(int type) { return type_kbits(type) ? 2 + 4 * type_kbits(type) : 0; }

struct Params {
    int iters = 3;        ///< scale refinement rounds after the first search (path given d -> d given path)
    int ncand = 1;        ///< tail-biting start-group candidates (rotations 16, 8, 24, 12, 20, 4, 28 ...); 1 = QTIP's
    int exact = 0;        ///< 1: try every start group (the exact tail-biting optimum at the current scale; slow)
    int scale_tries = 1;  ///< initial scales d0 * {1, 0.9, 1.1, 0.8, 1.2, ...}: the best of the first n searches
    uint16_t fixed_scale = 0;   ///< nonzero: this fp16 scale is used as is (no refinement); tests, planted solutions
};

struct BlockResult {
    double err = 0;       ///< sum_i h_i (w_i - rec_i)^2 for the stored block (h = 1 without importance weights)
    float scale = 0;      ///< the stored fp16 scale
    int searches = 0;     ///< Viterbi passes run
    int status = 0;       ///< 0 ok, 1 zero block (all weights zero or below fp16 resolution), 2 nonfinite input
};

/// The codebook entry of a state as the decoder sees it: four floats (int16 table / 8192, exact).
const float* codebook_state(int type, uint32_t state);   // 4 floats
const int16_t* codebook_table(int type);                // 32768 * 4 int16, the decoder's tq_state_table()
/// The 2048 codebook points as the decoder sees them (x, y interleaved): round(fp16 * 8192) / 8192.  The GPU encoder
/// uploads this table; it must be built by the host compiler (ggml-common-tq.h makes its table __device__ under hipcc).
const float* point_table();                     // 4096 floats

/// The state of step t (0..31) of a block's stream, by the decoder's rule, written bit by bit (independent of
/// tq_states.hpp).  qs = the 4K stream bytes.
uint32_t block_state(int type, const uint8_t* qs, int t);

/// Decodes `nblocks` blocks (block_bytes(type) each) to 128 floats each: d * codebook, d the fp16 scale.
void decode_blocks(int type, const uint8_t* src, int64_t nblocks, float* out);

/// Encodes one block of 128 weights (importance weights h[128] or null) into block_bytes(type) bytes.
BlockResult encode_block(int type, const float* w, const float* h, uint8_t* dst, const Params& p);

/// Encodes nblocks consecutive blocks on `nthreads` threads (0: hardware concurrency).  err_out (nblocks doubles) and
/// the returned sum of block errors are the objective sum h (w - rec)^2.  Sets *nonfinite to the number of blocks
/// with a NaN/Inf input weight (those are written as zero blocks).
double encode_blocks(int type, const float* w, const float* h, int64_t nblocks, uint8_t* dst, const Params& p,
                     int nthreads, double* err_out, int64_t* nonfinite, int64_t* zero_blocks);

/// TEST HOOK: one Viterbi pass over a scaled target (T[p][4] = weights / d of processing position p, H[p][4]) and the
/// legal-path states by position.  rot = first position processed, fixed_c = tail-biting group (-1 free).
float debug_viterbi(int type, const float (*T)[4], const float (*H)[4], int rot, int fixed_c, uint16_t* st);
/// Group sizes: predecessor groups G of a type; processing position p -> weight index of component j.
int debug_groups(int type);
int debug_widx(int type, int p, int j);
/// the stream bytes (4K) of a state path by position, as the encoder packs them
void debug_pack(int type, const uint16_t* st, uint8_t* qs);

// fp16 helpers (round to nearest even, like the hardware; exposed for the tests)
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

}  // namespace tqenc
