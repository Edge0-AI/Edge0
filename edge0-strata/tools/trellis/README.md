# Trellis encoder (research, branch f-dsv41-enc)

An encoder for the trellis expert formats that Strata already decodes (TQ2_T, TQK6, TQK7 of agentionai's llama.cpp fork)
and for two experimental lower rates, K5 (22 bytes per 128 weights, 1.375 bpw) and K4 (18 bytes, 1.125 bpw), in two versions:
TQK5X / TQK4X (ids 224 / 225, experimental version 2) use a computed codebook, two LCG words per state masked into four fp16
(`src/kernels/cpu/tq_computed.hpp`, from the z440 lab's "3single"), and TQK5T / TQK4T (226 / 227, version 1) the fork's
hashed point table. On real DeepSeek expert tensors the computed code costs 0.02 to 0.04 dB (K5 7.07 against 7.10 dB, K4 5.61
against 5.63) and needs no table lookups in a GEMV. The fork's types keep their table.
Written for the DeepSeek-V4.1-Flash compression plan (gate 2 and the start of gate 3). Nothing here changes an existing
type, layout or default; the engine's CPU decoder only gains K = 5 and 4 (ids 224 / 225, experimental version 1).

| file | what |
| --- | --- |
| `tqenc.hpp`, `tqenc.cpp` | the encoder: tail-biting Viterbi on the 2^15-state shift trellis, fp16 scale refinement, importance weights; a C API |
| `tqenc_test.cpp` | the tests (about 64,000 checks): streams and the circular wrap, Viterbi against a naive DP, planted solutions, bit-exact decode against `tq_states.hpp` and the scalar / AVX2 / AVX-512 kernels, special blocks, thread determinism, quality against the rate-distortion limit |
| `tqenc_hip.cpp`, `tqenc_hip_test.cpp` | the same search on a GPU (HIP, gfx1201): bit-identical blocks, about 20 M weights/s on one R9700 |
| `../tqenc.py` | Python wrapper (ctypes): `encode`, `decode`, the input-only block-128 Hadamard, signs, the `.tqx` container |
| `pilot.py`, `pilot_summary.py` | the block-quality pilot on a source GGUF (local files or HTTP range requests) |
| `gyro_reencode.py`, `gyro_vs_original.py`, `gyro_error_profile.py`, `hf_slice.py` | checks against agentionai's Gyro files and the original Qwen weights |

Build and test (Linux x86-64, g++ with AVX2; the GPU part needs ROCm):

    sh tools/trellis/build.sh test && build-tqenc/tqenc_test          # --quick skips the quality table
    sh tools/trellis/build.sh lib  && python3 tools/trellis/test_tqenc_py.py
    sh tools/trellis/build.sh hip  && HIP_VISIBLE_DEVICES=3 build-tqenc/tqenc_hip_test

`build.sh hip` compiles `tqenc.cpp` with g++ on purpose: under hipcc, `ggml-common-tq.h` turns its codebook into a
`__device__` variable and the host side reads zeros.

## Format facts the encoder relies on (checked against the decoder, not the comments)

* A block is `{fp16 d; 4K bytes}` for 128 weights: 32 steps, step t gives weights 4t..4t+3.
* State of step t: 15 bits. TQK: the 16-bit window of the stream at bit `(31 - t) K` (LSB first, circular modulo 32 K bits), masked
  to 15 bits. TQ2_T: `(qs[(t+31)%32] << 8 | qs[t]) & 0x7fff`.
* Weights of a step: `x = state * 0x9e3779b1`, the two codebook points `lut[x >> 21]` and `lut[(x >> 10) & 2047]`, times d. The
  decoder's table is `round(fp16 * 8192)`; the encoder uses that table, so its reconstruction is the decoder's, bit for bit.
* A legal stream is any byte string (every state is a window of the same circular stream); tail-biting only matters to the search.
* K5 and K4 are the TQK rule with K = 5 and 4. Nothing else changes (hash, codebook, mask, bit order, scale).
