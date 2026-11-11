# V100 (sm_70): three decode kernels that were built for other cards (2026-10-07)

On a V100 a verify window's GPU work is mostly memory-bound GEMVs that ran at a third to a half of the card's HBM2
bandwidth. Three kernels Strata already has for other cards are now Volta's defaults, each with exactly the old
arithmetic, so every output is bitwise the old one:

| What | Was on sm_70 | Now on sm_70 | Switch (old path) |
| --- | --- | --- | --- |
| Routed experts in VRAM (`native_expert_grouped`) | the CUDA layout (mode 0) | gfx906's mode 8: the codebook grid and the group's q8_1 activations in shared memory, SwiGLU + q8_1 in the gate/up epilogue; IQ2_XXS gate/up added to it (CUDA only) | `STRATA_EXP_MODE=0` |
| Dense 2-4 column GEMVs and the head (`native_mmvq_il`) | `native_mmvq` (the interleaved kernel was sm_80+) | the interleaved kernel with a rows table read off a V100 | `STRATA_MMVQ_IL=0` |
| Hyper-connection read, up projection | `gr_up_multi_kernel` | gfx906's `gr_up_fast_kernel` (built for CUDA, now also writes the QFUSE q8_1 tail) | `STRATA_GR_FAST=0` |

Every other card keeps what it ran: the new paths are picked only when the device is sm_70 (and gfx906 keeps its own
defaults). The idea of putting the lattice codebook in shared memory on Volta comes from 1Cat-vLLM's SM70 GGUF kernels
(`csrc/sm70_turbomind/ops/gguf_dp4a.cuh`); Strata's gfx906 kernels already did the same, so no 1Cat code was copied.

## Rig

- Tesla V100-SXM2-32GB (sm_70, PCIe gen3 x16), driver 580.159, a rented container (vast.ai): 2x Xeon Gold 6130, a
  cgroup quota of ~15 CPUs, 180 GB RAM. CUDA 12.8, GCC 11.4, Ubuntu 22.04.
- Engine: `main` at `e8ca9af` (0.1.40.2) with this change, `-DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`.
- Model: setup's IQ2_XS (`./setup.sh --cuda 12 --build --model IQ2_XS`), setup's config (`--spec 4 --kv int8
  --kv-resident 32768 --max-context 131072 --expert-cache auto`) plus `--pool-workers 14` (the container's quota; with
  setup's default of 32 workers the pool is throttled and decode halves).

## Kernels (synthetic or real rows, no engine)

**Experts**, `native_expert_bench <shard1> <layers> 28 T 0 8` (28 groups = a verify window's VRAM call, the model's
real rows; mode 0 vs mode 8, bitwise equal on all 48 layers):

| Layer gate/up / down | T = 1 | T = 3 |
| --- | ---: | ---: |
| IQ2_S / Q2_0 (34 layers, e.g. 24) | 118.5 -> 104.8 us | 180.4 -> 150.4 us |
| IQ2_XXS / Q2_0 (11 layers, e.g. 10) | 117.6 -> 91.4 us | 169.9 -> 134.1 us |
| IQ1_M / Q2_0 (3 layers; gate/up keeps the CUDA layout) | | 159.0 -> 144.7 us |

A strided call (the window's PCIe call, `grid_groups` 1..cap) keeps the CUDA layout: the AMD layouts launch a block
row per possible group, which cost the empty call 11.3 vs 6.5 us. A format without a shared-memory kernel falls back
to the CUDA layout on CUDA, not to R2 (R2 on IQ2_XXS gate/up: 183 vs 113 us). The nib_mask sign handling gfx906 uses
was also the faster one here (158.5 vs 162.2 us on `native_grouped_parity --bench`'s VRAM call).

**Dense GEMVs**, `mmvq_il_parity --bench` (two runs, mean; `il_rows1/2/4` all bitwise `native_mmvq`'s output), a few
of the cells the table picks (3 columns):

| Shape | `native_mmvq` | interleaved |
| --- | ---: | ---: |
| Q5_K head 2560 x 248,320 | 1,094 us | 821 us (rows 2) |
| Q6_K 2560 x 12,288 | 68.9 us | 47.9 us (rows 2) |
| Q4_K 2560 x 10,240 | 43.9 us | 32.5 us (rows 2) |
| IQ4_XS 2560 x 6,144 | 28.2 us | 19.3 us (rows 2) |
| Q6_K 2560 x 512 | 7.0 us | 9.2 us (rows 2): the table keeps `native_mmvq` below 2,048 rows |

**Hyper-connection read**, `fused_gr_bench` (bitwise): the fast norm/up take 4-6 us off a 51-72 us read in the
plain variant. The engine runs the staged variant, whose norm is the split one, so only the up projection changes
there (below).

## Engine: GPU time per verify window

`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`, `tools/ab_engine.py`: one server start per arm and round, the arms'
order alternating, 3 rounds; per round three short chats (256 tokens) and one 6,345-token prompt (122-128 tokens).
`orig` is `e8ca9af` as setup built it, `off` this branch with the three switches above set to the old paths, `v70`
this branch. Medians over the rounds of the engine's own stage table (ms per window, GDN + QSA layers):

| Stage | orig | off | v70 |
| --- | ---: | ---: | ---: |
| VRAM hits (routed experts) | 4.50 | 4.44 | 3.73 |
| q8 + qkv / q-idx GEMV | 1.60 | 1.57 | 1.40 |
| z | 1.03 | 1.02 | 0.87 |
| head | 0.99 | 0.97 | 0.69 |
| hc-read0 up | 1.08 | 1.08 | 0.96 |
| q + q-idx | 0.76 | 0.75 | 0.68 |
| out-proj | 1.35 | 1.34 | 1.40 |
| everything else on the GPU | 11.49 | 11.49 | 11.51 |
| **GPU work, the waits (A, B, CPU) left out** | **22.80** | **22.66** | **21.24 (-6.8%)** |

Decode tokens per second over the same runs (medians):

| Request | orig | off | v70 |
| --- | ---: | ---: | ---: |
| chat 0 | 65.8 | 67.1 | 74.7 |
| chat 1 | 55.0 | 58.1 | 62.7 |
| chat 2 | 57.1 | 58.0 | 59.7 |
| 6,345-token prompt | 52.3 | 48.7 | 52.5 |

**These decode numbers are noisy and mostly CPU-bound on this rig**: the windows' `waitCPU` (the GPU waiting for the
CPU experts) was 0.3 ms in some server starts and 6-9 ms in others, whatever the arm, which moves decode by 25-30%.
The GPU-side table above is the reliable comparison; on a PC whose CPU keeps up, the window is shorter by about the
1.5 ms it saves. Prompt speed (1,360-1,390 tok/s) does not change: the prompt path uses none of these kernels.

## Output checks

Each change keeps the old sums in the old order, so the engine's output is unchanged by construction; checked with:

- `native_expert_bench` on all 48 layers of the IQ2_XS shard, T = 1-4: mode 8 bitwise mode 0.
- `native_grouped_parity` (every gate/up x down format, window shapes, `STRATA_EXP_MODE` 0, 2, 7, 8): 0 failures.
- `mmvq_il_parity`: OK (every case, T 2-4, rows 1/2/4 and the table, bitwise); `mmvq_multi_parity`: ok.
- `fused_gr_bench` (fast vs old, bitwise), `gr_multi_parity` OK, `gr_parity` 0 failures.
- The whole `ctest` on the V100 (`-DSTRATA_BUILD_TESTS=ON`): 124 passed, 2 skipped (`prefill_fused_*`, sm_80+), 3 failed
  for this rig, not the change: `expert_multi_test` (the CPU has no AVX512-VNNI), `platform_memory_test` (`mlock` in
  the container), `ple_parity` (wants the Q2_0 pack's second shard; this rig has IQ2_XS).

Raw: `stage-profile.txt` (the engine's stage lines of every run above), `mmvq-il-bench-v100.txt` (both bench runs).
