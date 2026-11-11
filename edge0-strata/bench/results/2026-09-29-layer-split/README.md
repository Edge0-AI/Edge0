# Layer split across GPUs (2026-09-29)

**Rig (a community member's PC, "RIBPC"):**
- Ryzen 9 9950X3D, 62 GB DDR5, Windows 11, CUDA 13.1.
- RTX 5080 16 GB (x16), RTX 3090 24 GB (x4 gen4, ~6 GB/s), RTX 2080 Ti 22 GB (x4 gen3, Turing sm_75; the
  experimental `STRATA_EXPERIMENTAL_SM75` build).

**Model:** the Coder (GSQ-RCO IQ1_M, 48 layers x 256 experts = 12,288 expert pairs, ~23 GB of experts), 32K
context, int8 KV, `--prefill auto`, `--spec 4`. It is the only model that fits the rig's RAM and disk.

**Tools:** `rib_pp.py` (needle prompts at 2.7K/8K/16K/28K tokens with a code word at 50% depth, a 2-turn chat that
must reuse its checkpoint, greedy decode of a story and a code prompt) and `rib_norepeat.py` (the Phase A
no-repeat and sanity cases). Raw results are in `data/`.

## Correctness

| Check | Result |
|---|---|
| This branch vs the 0.1.20 release, one GPU (5080), fixed cache, all 10 cases | **10/10 byte-identical** |
| This branch, one GPU vs the split on the same GPU (K=24, `--split-device 0`) | **10/10 byte-identical**: the hand-off is exact |
| 5080 + 3090 (auto) and 5080 + 2080 Ti + 3090 (auto): no-repeat under presence 100 (greedy and sampled), 144 sanity | all pass, 0 repeats |
| Needles at 2.7K-28K tokens, every configuration | all found |
| 2-turn chat (the checkpoint restored on every GPU) | reused 6,631-6,632 of 6,660 tokens, answer right |

Across GPUs the text is not byte-identical to one GPU: experts the CPU computed on one card run on a GPU in the split,
and they round differently (the same effect as `--expert-cache`, bench/results/2026-09-27-cache-parity).

## Speed

**Prompt tok/s** at 16K / 28K tokens, and **decode tok/s** for story / code (greedy; three repeats averaged
in the K sweep):

| Configuration | Split | Prompt 16K | Prompt 28K | Decode story | Decode code | Decode hits |
|---|---|---|---|---|---|---|
| 0.1.20, 5080 alone | - | 2,045 | 2,005 | 83.2 | 88.2 | 70-84% |
| this branch, 5080 alone | - | 2,017 | 1,974 | 83.1 | 88.0 | 70-84% |
| this branch, 5080 alone, 3 repeats (adaptive swaps learn the prompt) | - | 1,726 | 1,970 | 86.7 | 105.2 | 91-92% |
| 3090 alone | - | 1,253 | 1,211 | 71.7 | 92.9 | 86-93% |
| 5080 + 3090 | K=18 | 1,547 | 1,694 | 77.2 | 103.8 | 99.9% |
| 5080 + 3090 | K=22 | 1,840 | 2,074 | 80.2 | 106.4 | 99.8% |
| 5080 + 3090 | K=24 | 1,978 | 2,258 | 83.5 | 109.5 | 99.5% |
| **5080 + 3090** | **K=26** | **2,039** | **2,357** | **83.8** | **109.7** | 99.3% |
| 5080 + 3090 | K=28 | 2,011 | 2,287 | 83.5 | 112.5 | 99.0% |
| 5080 + 3090 | K=30 | 1,935 | 2,218 | 81.1 | 109.6 | 98% |
| 5080 + 3090 | K=32 | 1,818 | 2,063 | 83.0 | 108.7 | 98% |
| 5080 + 3090 | auto (K=22) | 2,037 | 2,073 | 79.5 | 108.9 | 99.7% |
| 3090 + 5080 | auto (K=30) | 1,672 | 1,876 | 90.3 | 105.1 | 99.4% |
| 5080 + 2080 Ti + 3090 | auto (22, 23) | 1,687 | 1,847 | 67.6 | 89.6 | 99.7% |
| 5080 + 3090 + 2080 Ti | first auto (12, 33) | 1,148 | 1,144 | 51.9 | 69.7 | 100% |
| 5080 + 2080 Ti | first auto | 914 | 906 | 45.5 | 58.0 | 97% |

What decides it:

- **The best two-GPU split for the Coder is K=26-28.** Prompts run +18-20% over the 5080 alone at 16-28K;
  decode is on par for story and +4-7% for code.
- **Residency saturates quickly: the per-layer GPU speed decides.** From K=18 up, both caches hold 98-100% of
  the routed experts. Moving layers to the faster card is what helps: the 3090 costs 0.50 ms per layer per
  window, the 5080 0.33 (`STRATA_SPLIT_TIMING`, the "wait for the GPU" per stage).
- **Decode is bounded by that GPU time.** One 5080 with its cache adapted spends 16.3 ms waiting for its GPU and
  5.5 ms in the CPU pool per window. The split removes the pool (0.8-1.0 ms) but adds the slower card's layers:
  7.9 + 12.1 ms at K=24. The per-layer GPU half (kernels plus the host round trip that publishes each layer's
  expert plan) is the next thing to cut, for one GPU as well (perf-review-plan E-6).
- **A third, slower card costs more than it holds.** The 2080 Ti's one-layer stage took 3.1 ms per window.
  For a model that already fits two cards, leave it out. It pays only when the two faster cards cannot hold most
  of the routed experts (a bigger model; not testable on this rig's 62 GB of RAM).
- **Prompt chunk:** 1024 instead of 2048 frees ~0.7 GB per card, but prompts read 7-12% slower.

## Auto placement

`--layer-split auto` predicts one decode window's time for every placement (every placement up to three GPUs) and
keeps the fastest:

- **per-layer time** from SMs x clock (0.33 ms for an RTX 5080);
- **cache contents:** each card's layers' profiled pairs, hottest first, as the fill does;
- **routed mass** of rank r taken as (r+1)^-1.2 (fits the sweep's hit rates within 0.1-1%);
- **missed mass** costing 190 ms per unit (fitted to the sweep's best K; `STRATA_SPLIT_MISS_MS` overrides).

It picked K=22 on the 5080 + 3090: within 0-12% of the best fixed split on prompts and on par on decode. It gives
a slow third card a single layer.

## Reproduce

On a rig: the `strata-remote` tools, then `rib_matrix.bat`, `rib_ksweep.bat`, `rib_validate.bat` (in the dev
notes, not in this repository).
