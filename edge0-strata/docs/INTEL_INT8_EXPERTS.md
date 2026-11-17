# Fused int8 DPAS expert products on Battlemage (branch f-151-b70)

Behaviour note, written before the kernel (2026-10-18) and kept in step with `sycl/src/prefill/esimd_moe.dp.cpp`. The measured numbers
are at the end.

## What it replaces

The prompt path computes a routed expert as dequantize (gate/up and down to FP16, 9.8 MB written) then oneMKL GEMMs and SwiGLU. On
the Arc Pro B70 (IQ3_S pack, 4K prompt) that chain is ~38 us of GPU time per expert (dequant 25.6 us, GEMMs and SwiGLU ~12 us) and
23% of the prompt. The fused path decodes the quantized weights in registers straight into the DPAS B operand, so the FP16 copy
never exists and the weights are read once, compressed (2.1 MB per expert instead of 9.8 MB written and read back).

## Switches

* `STRATA_PREFILL_MMQ=1` (alias `STRATA_PF_I8=1`): every group of experts whose formats are supported takes the int8 chain.
* `STRATA_PREFILL_MMQ=0`: the FP16 path everywhere. The name is MMQ's because the other backends' int8
  prompt-expert path is MMQ; here it is this kernel.
* `STRATA_PREFILL_MMQ=auto` (the Battlemage default): per group, int8 while the group's mean rows per expert are under the crossover measured on this GPU.
  At the first layer of each expert format the engine times both chains on synthetic experts (random blobs of the layer's format and
  shape, uneven row counts like a real group) at 16, 32, 64, 128 and 256 mean rows per expert, interpolates where the FP16 chain
  becomes the faster one, flags the result "clear" when the cost ratio falls steadily and the bracketing samples are 5% away from
  1, and keeps it in `~/.cache/strata/i8-crossover.txt` (`STRATA_CACHE_DIR` moves it) keyed by device name, driver version, the two
  ggml types and the shape. A matching key is read back instead of measured. The measurement costs ~1.2 s once, during the first
  prompt. A measurement that fails (no VRAM) falls back to 64 rows.
* `STRATA_I8_FORCE=0` hides the device from the kernels, `STRATA_I8_MT=2|4|8` forces the tile height, `STRATA_I8_WG=0` the
  non-work-group kernels, `STRATA_I8_GRF=128` the 128-register mode, `STRATA_I8_DBG` bring-up switches (skip decode / loads / stores).
* Default (unset): `auto` on a Battlemage GPU (bmg-g21 or bmg-g31 by the SYCL architecture id, not by name; the kernels must be in the binary and pass `available()`), the FP16 path on every other GPU. The engine prints one line at load ("int8 prompt experts are on ... STRATA_PREFILL_MMQ=0 turns them off"). If the kernels are not in the binary for the device (an AOT build for other devices) or a launch throws, the group and the rest of the process take the FP16 chain with one log line. The activations are quantized, so the bits differ from the FP16 path (see "Quality").

## Tiles and thresholds

* One hardware thread (ESIMD, sub-group 16, 256 GRF) owns 16 output rows of one expert, all of K, and `MT x 8` activation rows.
  `MT` = 2, 4 or 8 from the group's mean number of 8-row tiles per expert (<=2 -> 2, <=4 -> 4, else 8). A thread's accumulators are
  MT tiles of 512 bytes in an 8 KB register file (256 GRF of 32 bytes), so MT 12 or two output tiles per thread spilled 3-15 KB and ran
  2-5x slower. An expert with more rows than 8 MT is split over row-block threads, each decoding the weights again.
* A work group is 8 such threads on 8 consecutive output tiles (128 outputs) of one row block: the activation tiles of the row block
  are read from global memory once per group, in chunks of 4 steps of 32 (128 of K), into double-buffered local memory (one barrier per
  chunk), and every thread reads them from there; the group also copies the codebook the decoders gather from into local memory.
  IQ2_S gate/up keeps the plain kernel: its 8 KB codebook costs more to copy per group than the gathers save (35 against 16 us).
* K step 32 (one s8 DPAS 8x16x32): decode the 16 rows' 32 values (lanes = rows, one weight decode per tile, applied to all MT row
  tiles), one DPAS into a zeroed int32 tile, float epilogue. Group of up to 16 experts per launch (the same group the FP16 grouped path
  holds in the prompt walk), 4 launches per group (quantize, gate/up, SwiGLU + quantize, down) against 4 per expert.
* The kernels need the s8 8x16x32 matrix combination (Xe2: Battlemage, Lunar Lake). Large GRF: `grf_size<256>` is requested for
  every kernel; a device that cannot build it keeps its default and the spills show in the build log. DP4a fallback: none; without the
  DPAS combination `available()` is false and the FP16 path runs (the build without ESIMD for an Alchemist / Xe-HPC AOT device
  compiles stubs, `STRATA_I8_NO_ESIMD`).

## Scales

* Activations: llama.cpp's q8_1 grid. Per row and 32 of K, `d = amax / 127`, `q = round(x / d)` half away from zero, int8 stored in
  DPAS A-operand order (`[8-row tile][K/32][8 rows x 32 bytes]`, experts padded to 8 rows with zeros), the 8 row scales of a tile
  contiguous.
* Weights: IQ3_S `d (1 + 2 s)` per 32, IQ3_XXS `d (2 s + 1) / 4`, IQ2_S `d (2 s + 1) / 8` per 16 (two DPAS per 32, the other half
  zeroed), IQ4_NL `d` per 32, Q2_0 `d` per 64; the int8 values are the codebook entries with their signs (IQ3_S/IQ3_XXS/IQ2_S: 1..62,
  IQ4_NL: -127..113, Q2_0: -1..2), never rounded.
* Each 32-block's int32 dot product (<= 32 * 127 * 127) is exact and exactly convertible to float; it is scaled by the weight scale of the
  column and the activation scale of the row and added to a float accumulator, K ascending (the float multiply and add are the only
  roundings besides the activation quantization). After gate/up, `silu(gate) * up` (the FP16 path's formula, in float, not rounded
  to FP16) is quantized per 32 of the 640 FF values into the down input.

## Edge cases and fallback rules

* A row count is padded to a tile of 8 with zero rows (stores are masked to the real rows); an expert with no rows is not in the group.
* IQ3_S rows are 110-byte blocks (IQ3_XXS 98, IQ2_S 82, IQ4_NL / Q2_0 18): 2-byte aligned only, so every field is read as dwords, two
  gathers and a funnel shift when not dword aligned. The down matrix ends the expert blob, so its reads clamp the second dword of an unaligned read to the
  matrix's last dword (`ld32c`); the gate/up reads stay inside the blob (the down matrix follows them). The first version read up to 3 bytes
  past the end of a test allocation and faulted the GPU (an xe CAT error and an engine reset; nothing else was affected).
* Supported: gate/up IQ3_S (21), IQ3_XXS (18), IQ2_S (22) with K a multiple of 256; down IQ4_NL (20) with K a multiple of 32 and Q2_0
  (42) with K a multiple of 64; 640 FF x 2560 embedding; output rows a multiple of 16; a held group of at most 16 experts. Any other type or
  shape makes the layer take the FP16 path (IQ4_XS gate/up, 1 of 48 layers of the IQ3_S pack). A group the kernels cannot run (no VRAM for
  the int8 scratch, a format they refuse) takes the FP16 chain for that group.
* Multi-GPU: the codebooks, the scratch and the availability are kept per device.

## ESIMD on this compiler (oneAPI 2026.1, bmg-g31) - what went wrong while writing it

* A dynamic index into a kernel-argument array gave a garbage value (the first weight read faulted the GPU, the xe driver logged
  hundreds of page faults and reset the engine, the box recovered each time): the kernels read the arrays at constant indices
  (`pick_expert`, the unrolled expert lookup in the quantize kernels).
* `t % n` and `t / n` with a run-time `n` gave `t % n == t`: the kernels use 2-D grids.
* `gather<T, N, VS>` with VS > 1 is not available on Battlemage; a lambda capturing a simd by reference puts it in private memory.

## Quality

The int8 path is near-lossless against the FP16 path. 28 prompts (code, reasoning, chat, multi-turn, long up to 3.2K tokens), 6002
positions teacher-forced on the FP16 path's greedy continuations (`cuda_kl.py` with `STRATA_FOLLOW`), fixed flags: reference against
itself KL exactly 0; int8 against reference KL mean 0.0002 nats (p99 0.006, max 0.04), top-1 99.7%, perplexity ratio 1.0004, first-token
KL 0.0005. For comparison a chunk-size change (`--prefill 8192` against 4096) moves the first-token distribution by 0.01-0.03 nats.
The kernel arithmetic itself matches an exact host reference on the same int8 activations to 3e-7 relative; against FP64 with the
unquantized activations each product carries ~1% rms (the q8_1 grid; synthetic data with 1% outliers), the chain 1.7% against
dequantize + oneMKL. `esimd_moe_parity` runs all six gate/up x down type pairs.
