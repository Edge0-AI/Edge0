# QSA_ROWS: the decode window's QSA block scores with the queries in registers (`STRATA_QSA_ROWS=1`, opt-in)

Opt-in: `STRATA_QSA_ROWS=1` (exactly `1`). Unset the engine runs the kernels it ran before. CUDA, the verify window's
decode (calls of `qsa_block_scores` without `active_blocks` and with up to 4 queries); the prompt path and HIP are not
touched. Scores and top-k ids are bitwise the multi kernel's (`qsa_rows_bench --rows` checks it).

## Why

In a 131k decode the QSA kernel that grows most with the window's rows is `block_scores_multi_kernel`: about 20 us more
per call for every row of the window on an RTX 2080 Ti and an RTX 3060 (nsys; 8 QSA layers on the second card of a
layer split, 4 on the first). The keys are already read once for all the queries; the cost per row is in the shared
memory and shuffle pipes: per (key block, query) the kernel reads the four query slices from shared memory (4 x
`LDS.128`) and reduces each head's lane products with its own five shuffles (20 per block and query): at 32 769 blocks
that is 15-20 us of issue per query.

## What it changes

`block_scores_rows_kernel<NQ>` (`src/kernels/cuda/qsa_select.cu`, NQ = 1..4):

1. the NQ queries' slices (4 heads x 4 floats per lane) stay in registers for the whole kernel (44 / 63 / 78 / 94
   registers, no stack, `cuobjdump -res-usage`);
2. the lane products of the 4 heads are reduced with a scatter that keeps `warp_sum`'s pairings: xor 16 and 8 leave
   one head per lane, then 4, 2, 1 - six shuffles for the four heads instead of twenty - and gathered on lane 0 in head
   order (three shuffles).

Every addition adds the same two operands of the same `warp_sum` level (float addition commutes), the product
expression is the same text in the same file, the relus are summed in head order: the same bits as the multi kernel
and the per-query kernel. Tail block and the `+1e9` as before. Beyond 4 queries, with `active_blocks`, or without the
variable, the multi kernel runs.

## Micro-bench: `qsa_rows_bench`

Not in the default build: `cmake --build <dir> --target qsa_rows_bench`. Synthetic data at the real geometry (24 query
heads x 256, 2 KV heads, indexer 4 x 128, blocks of 4 cells, top-k 2048), int8 pools, no model.

```
qsa_rows_bench [--device N] [--iters 100] [--ctx 131072] [--max-context 262144] [--rows] [--no-dense]
```

Without `--rows` it measures what the T rows of a verification window cost in a QSA layer's decode kernels (block
scores, top-k, attention over each row's selection, indexer append; `--no-dense` skips the drafter's 32768-cell window).
`--rows` runs `block_scores_rows_kernel` against the multi kernel - scores and top-k ids bit for bit (memcmp) at T =
1..4 at 131k and at a short context (cells 100-103, a full tail block), 16 cases per card - and prints both kernels'
microseconds. Exit 0 (77: no GPU; 1: a parity failure).

`qsa_block_scores`, us per call, 131k context (both kernels read the same pooled keys):

| | T=1 | T=2 | T=3 | T=4 | per row |
|---|---|---|---|---|---|
| RTX 2080 Ti, multi (reference) | 47.2 | 56.9 | 78.2 | 101.2 | +18.0 |
| RTX 2080 Ti, `STRATA_QSA_ROWS=1` | 36.2 | 38.0 | 49.2 | 54.6 | +6.1 |
| RTX 3060, multi | 73.4 | 109.4 | 151.9 | 193.5 | +40.0 |
| RTX 3060, `STRATA_QSA_ROWS=1` | 63.3 | 75.5 | 98.0 | 130.2 | +22.3 |

Parity: 8 cases per card, 0 differ. On the RTX 3060 at T=4 the
kernel is at 94 registers (2 CTAs per SM); a two-pass version of 2 queries would stay at 63 if that card matters.

## In the engine

Layer split RTX 3060 (layers 0-15) + RTX 2080 Ti (layers 16-47, head), Windows, `--pipeline-windows 2`, 131k context:
per window the second card saves -0.09 ms at T=1 and -0.37 ms at T=4 (8 QSA layers), the first -0.04 / -0.25 (4
layers). End to end on a deterministic 131k bench (forced tokens and window sizes, 1024 tokens, B-A-B): **+3.2% and
+0.1%** decode tokens/s in the two B arms, the text identical.

## Not done

A multi-row attention kernel (K/V of a cell read once for all the rows that selected it) is possible bit for bit only
if every row keeps its own chunk partition; at the decode window's sizes (cap 2051 cells, T <= 4) the blocks of all rows
fit in one wave and the kernel is bound by its block's latency chain, so the gain is bounded by ~0.1-0.2 ms per row on
8 layers: not worth it here. `STRATA_QSA_ROWS` turns on the scores kernel only.
