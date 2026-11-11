# V100 (sm_70): the prompt's experts on FP16 tensor cores (2026-10-07)

On sm_70 the prompt path ran the routed experts on llama.cpp's MMQ, whose integer products are `dp4a` there (Volta has
no int8 tensor cores), and IQ1_M layers on the FP16 path (each expert dequantized to global memory, then one cuBLAS
call per product). `gemm_iq_f16_grouped` (`src/kernels/cuda/iq_kernels.cu`) runs a whole MMQ expert group per launch on
the FP16 tensor cores (WMMA, FP32 sums): a block takes 128 weight rows of one expert and up to 256 of its token rows,
dequantizes each weight superblock once into shared memory with `dq_dispatch` (the FP16 path's own formulas), and
streams the FP16 activations through it, the next tile prefetched into registers. sm_70 takes it by default for the
layers MMQ would take and for IQ1_M (which then rides MMQ's grouping); `STRATA_PF_WMMA=0` is the old path,
`STRATA_PF_WMMA=1` turns it on for any CUDA card (not measured elsewhere).

## Rig

As in [2026-10-07-v100-decode-kernels](../2026-10-07-v100-decode-kernels/README.md): V100-SXM2-32GB (PCIe gen3 x16),
2x Xeon Gold 6130 in a container with a ~15-CPU quota, CUDA 12.8; setup's IQ2_XS (gate/up IQ2_S on 34 layers, IQ2_XXS
on 11, IQ1_M on 3; down Q2_0), setup's config (`--prefill auto`: 8,192-token chunks here) plus `--pool-workers 14`.

## Prompt speed

`tools/ab_engine.py`, 3 rounds, the arms' order alternating, one server start per arm and round; the server's own
prompt tokens/s. `orig`: `e8ca9af` as setup built it; `off`: this branch with `STRATA_PF_WMMA=0`; `new`: this branch.

| Prompt | orig | off | new | new vs orig |
| --- | ---: | ---: | ---: | ---: |
| 26,359 tokens (median of 3) | 1,467 tok/s | 1,462 | 1,608 | **+9.6%** |
| 6,345 tokens (median of 3) | 1,390 tok/s | 1,393 | 1,534 | **+10.3%** |

All runs: `ab-runs.jsonl` (the spread between rounds is under 1%). The engine's phase table
(`STRATA_PREFILL_TIMING=1`, the 26,352-token prompt, ms):

| | orig | new |
| --- | ---: | ---: |
| experts gate/up | 3,913 | 3,670 |
| experts down | 2,229 | 1,212 |
| dequant (the IQ1_M layers' FP16 path, and the rest of that phase) | 740 | 575 |
| GPU timeline | 17,391 | 15,847 |

Decode does not use this path (decode tok/s in the same runs: within the noise).

## Accuracy

- `pf_wmma_parity` (synthetic, no model): every format it takes (IQ2_XXS, IQ2_S, IQ1_M gate/up, 1280 x 2560; Q2_0 down,
  2560 x 640), 7 experts with 0-300 rows, against a double-precision product of the same FP16 weights and activations:
  max error 4.4e-4 to 5.2e-4 of outputs up to ~110 (Q2_0: 2.8e-6 of 1.16) - FP32 summation order.
- **Teacher-forced** (`STRATA_LOGPOS` with `STRATA_LOGPOS_TOPK=20`; a long random-word prompt read by the batched path,
  then a continuation read through the decode windows with `--short-read 1024`, each token's log-probability against
  the text). MMQ is not a reference either - its activations are q8_1, these are FP16 - so the table puts the change
  next to what MMQ alone does when only the chunk size changes:

| Continuation (after) | Compared | argmax agreement | KL (top 20) | perplexity |
| --- | --- | ---: | ---: | --- |
| 363 tokens of prose (26,359-token prompt) | MMQ vs this | 98.3% | 0.0083 | 50.7 / 60.5 |
| | MMQ 8,192 vs 32,768-token chunks | 98.3% | 0.0084 | 50.7 / 50.6 |
| | MMQ 8,192 vs 16,384-token chunks | 93.6% | 0.0177 | 50.7 / 39.2 |
| 327 tokens of Python (12,676-token prompt) | MMQ vs this | 94.5% | 0.0312 | 3.98 / 3.38 |
| | MMQ 8,192 vs 4,096-token chunks | 96.3% | 0.0253 | 3.98 / 3.49 |
| | MMQ 4,096 chunks vs this | 96.3% | 0.0170 | 3.49 / 3.38 |

  Two MMQ runs of the same configuration were bit for bit the same. After a long prompt these continuations move this
  much from any change of the prompt path's arithmetic (the perplexities swing both ways); the change sits inside
  what the chunk size alone does. Two texts are not a quality benchmark; a run against a full-precision reference
  (llama.cpp, as in [UNSLOTH_Q4](../../../docs/UNSLOTH_Q4.md)) was not made.

## Notes

- One observation from these runs, not followed up: on the 12,676-token prompt MMQ with `--prefill 4096` read
  1,560 tok/s against 1,370 with `auto` (8,192 here) - a single run each.
- Tried and dropped on the way: the FP16 path for chunks of 16K+ tokens (+10% at `--prefill auto:32768`, but superseded
  by this, which also wins at setup's chunk size); a row-per-warp hyper-connection down projection (no gain in the
  engine). Groups of 32 experts instead of 16: no gain.
