# RX 6900 XT (gfx1030): PR #540's attention kernel with 8 cells per step and DPP lane exchanges (2026-10-04)

`attn_chunk_kernel_pre75` (PR #540) is the default attention kernel with its 12-head reductions reorganised so that each
cell needs 16 lane exchanges instead of 60, bit for bit the same sums. It was built only into the experimental CUDA build
and used below sm_75. On HIP the default kernel ran, whose score loop is bound on gfx1030 by the LDS pipe: every
`__shfl_xor_sync` compiles to `ds_bpermute_b32`, which shares that pipe with the shared-memory q reads. This change builds
the pre75 kernel for HIP wave32 and runs it on gfx103x by default, with 8 cells per step (`NC_`, CUDA keeps 2) and the
lane exchanges in registers (`DPP_`: `row_xmask` within a 16-lane row, `v_permlanex16` across the two rows; CUDA keeps
`__shfl_xor_sync`). `STRATA_ATTN_PRE75=0` is the default kernel, so the main numbers below are one binary with and
without that switch.

## Rig

- One AMD Radeon RX 6900 XT 16 GB (gfx1030, PCIe 4.0 x8) of the two in the machine; the other card idle. AMD Ryzen 5
  5600X, 128 GB DDR4-3200.
- Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0.0 (HIP 7.15.26333, rocBLAS 5.6).
- Engine: `origin/main` at `6f32ec0` (0.1.39) with this change (`a649e42`), `-DSTRATA_ENABLE_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_PREFILL_MMQ=ON` (as setup builds it).
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (2 shards, `--native`), MTP draft layer on, `--kv int8 --kv-resident 32768
  --max-context 131072 --expert-cache auto`; [strata-1gpu.json](strata-1gpu.json) (`/m` the model directory, `/work`
  the build). Not one of setup's models.

## Why: the score loop on gfx1030

Compiler output for the INT8-KV kernels (`-Rpass-analysis=kernel-resource-usage` for registers and occupancy; the
assembly for the instruction counts):

| kernel | VGPRs | waves / SIMD | `ds_bpermute` per warp and 8 cells (score loop) |
| --- | ---: | ---: | ---: |
| default | 40 | 16 | 480 |
| pre75, NC 1 | 101 | 9 | 128 |
| pre75, NC 2 (CUDA's value) | 142 | 7 | 128 |
| pre75, NC 4 | 149 | 6 | 128 |
| pre75, NC 8 | 144 | 7 | 128 |
| **pre75, NC 8 + DPP (this change)** | 148 | 6 | **0** |

LDS per block is 15.5 KB and does not limit occupancy. The DPP variant's kernel has 103 LDS instructions where the NC 8
variant has 257 (static counts); its 154 `ds_bpermute` are gone. ROCm 10.0's hipcc does not turn an xor shuffle into
DPP on its own, so the exchanges are written with `__builtin_amdgcn_mov_dpp` (`row_xmask`, M = 1, 2, 4, 8) and
`__builtin_amdgcn_permlanex16` (M = 16), in the same pairing order as the shuffles; a 32-lane unit test of the exchange
primitives against `__shfl_xor` matched bit for bit before the kernel was changed.

## Prompt speed, this branch

Cold server per run; three chats whose first message is 9,427 / 34,659 / 105,811 tokens of this repository's docs and
sources, read by the batched prompt path; the server's own line `strata serve: prompt N tokens = 0 reused + N read in T
ms`. One run per arm (the two are the exactness runs below):

| prompt tokens | default kernel (`STRATA_ATTN_PRE75=0`) | this change | gain | read time |
| ---: | ---: | ---: | ---: | ---: |
| 9,427 | 440.7 tok/s | 459.1 tok/s | +4.2% | 21.4 s -> 20.5 s |
| 34,659 | 466.9 tok/s | 493.6 tok/s | +5.7% | 74.2 s -> 70.2 s |
| 105,811 | 460.4 tok/s | 488.3 tok/s | +6.1% | 229.8 s -> 216.7 s |

Attention is a small share of the prompt on this card while the 16-bit GEMMs run rocBLAS's slow fallback kernels (#835
changes that); the gain grows with the prompt because the attention's share does.

## Step by step, on the FP16 prompt path

With #835 applied the GEMMs are about twice as fast and attention is a larger share, so the kernel variants are easier to
tell apart. These runs are a test build of `6f32ec0` + #835 with the variants selectable by environment variables
(`STRATA_ATTN_PRE75=1`, `STRATA_ATTN_NC`, `STRATA_ATTN_DPP`), not the binary of this branch; same card, same three
chats, cold server per run, one run per row except NC 2 (two runs, 778.9 / 981.1 / 998.0 and 779.0 / 981.6 / 996.9 tok/s,
within 0.2%):

| attention kernel | 9,427 | 34,659 | 105,811 |
| --- | ---: | ---: | ---: |
| default | 746 | 917 | 928 |
| pre75, NC 1 | 763 (+2.3%) | 950 (+3.6%) | 960 (+3.4%) |
| pre75, NC 2 | 779 (+4.4%) | 982 (+7.0%) | 997 (+7.5%) |
| pre75, NC 4 | 787 (+5.5%) | 997 (+8.7%) | 1,008 (+8.6%) |
| pre75, NC 8 | 792 (+6.2%) | 1,006 (+9.7%) | 1,025 (+10.5%) |
| **pre75, NC 8 + DPP** | **801 (+7.4%)** | **1,022 (+11.4%)** | **1,041 (+12.2%)** |

Every step up in NC was faster although occupancy went down from NC 1 (9 to 6-7 waves per SIMD): on this card the q reads
and the lane exchanges on the LDS pipe bound the loop, not occupancy. That is the opposite of the V100, where #540 found
NC 4 slower than 2, which is why the template keeps CUDA at 2. NC can only be a divisor of 8 (a warp handles 8 cells), so
8 is the largest value as well as the fastest measured. The DPP variant on top is another +1.1 / +1.6 / +1.5%.

With both changes a prompt reads at 1.82x / 2.19x / 2.26x `6f32ec0` on this card (439 / 466 / 461 -> 801 / 1,022 /
1,041 tok/s). The two changes touch different files and either applies alone.

## Exactness

Teacher-forced log-probabilities (`STRATA_LOGPOS`, top 256, the method of `bench/results/2026-10-03-v100-prompt-attn`):
the three chats above, each with a fixed assistant reply and a last message of about 580 tokens read through the verify
windows, every position scored: 578 / 519 / 481 = 1,578 positions. This change against the default kernel, same binary:
KL 0 at every position, argmax and top-10 agreement 100%. The same held for every NC variant and for the DPP variant in
the step-by-step runs (each against its own default-kernel run on the FP16 base). The sums are the same values added in
the same pairing order; the kernel only moves them differently.

## Decode

The verify window runs the same attention kernel over its few rows. Measured on 2026-10-05, one card, this branch's binary,
greedy, 256-token cap (the answers stopped at 128-145 tokens) to a 2,267-token prompt and three of 3,906-4,919 tokens,
`STRATA_DECODE_TIMING=1`, two rounds alternating the arms with a fresh server each:

| prompt tokens | default kernel, ms / window (round 1 / 2) | this change | difference |
| ---: | ---: | ---: | ---: |
| 2,267 | 56.00 / 56.08 | 55.68 / 55.63 | -0.4 ms (-0.7%) |
| 4,380 | 42.14 / 42.19 | 41.88 / 41.89 | -0.3 ms (-0.7%) |
| 4,919 | 42.61 / 42.39 | 42.03 / 41.95 | -0.5 ms (-1.2%) |
| 3,906 | 44.17 / 44.21 | 43.59 / 43.67 | -0.6 ms (-1.3%) |

The window is 0.3-0.6 ms shorter with this change in both rounds, where the two rounds of one arm differ by at most
0.2 ms; decode 43-53 tok/s against 43-56. The greedy answers took the same number of verify windows in 7 of the 8
request pairs (the 3,906-token prompt once differed by two windows: adaptive expert swaps were on, so an expert can run
on the GPU in one run and on the CPU in the other, and the two round differently).

## Limits

One machine, one model (IQ3_S packed from a non-setup GGUF), one corpus; one run per arm on this branch. The step-by-step
series was measured on a test build with #835, not on this branch alone. CUDA builds were not run (no CUDA here): the
CUDA path is unchanged by construction, since the template defaults equal the old constants and `shfl_x<false>` is
`__shfl_xor_sync`. Other wave32 AMD cards (gfx1031 / gfx1032, RDNA3, RDNA4) were not run; `STRATA_ATTN_PRE75=1` turns
the kernel on there.
