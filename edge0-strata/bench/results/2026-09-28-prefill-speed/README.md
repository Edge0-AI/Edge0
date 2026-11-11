# Faster prompt reading (engine 0.1.13)

RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5. 32K-token prompt (`bench/prompts/long/32k-01-code-agent.ids`), one-shot
engine, `--expert-cache auto`, greedy. Numbers are prompt tokens/s.

## What changed, step by step

| Step | What | Result |
|---|---|---|
| Buffers | The attention half and the MoE half of a layer share one scratch region (~260 KB less per prompt token) | bit-identical; chunk 4096 borrows 1.78 GiB instead of 2.84 |
| PLE rows | The next chunk's PLE rows are read from the model file on a thread while the GPU works | bit-identical; 762 -> 790 at chunk 4096 |
| Chunk size | `--prefill auto`: the largest of 8192 / 6144 / 4096 / ... whose buffers fit in the cache slots it may borrow; a request borrows only what its own prompt needs | Q2_0: 4096 791, 6144 878, 8192 973 |
| Host copies | Experts the arena could not pin are copied to pinned buffers by helper threads instead of the launching thread | bit-identical; IQ3_S at 4096: 551 -> 652 |
| PLE block | Computed for the whole chunk (two GEMMs) instead of token by token | Q2_0 981 -> 1053, IQ3_S 831 -> 940 |
| Experts | llama.cpp's MMQ kernels (weights stay quantized, int8 tensor cores) instead of dequantize-to-FP16 + cuBLAS | Q2_0 1052 -> 1130, IQ3_S (8192) 932 -> 1014 |
| Streaming | At chunks of 2048+ every non-resident expert streams in a fixed order through a ring, so the next layer's arrive while the current layer's attention runs | bit-identical; Q2_0 1130 -> 1290, IQ3_S -> 1208 |

Tried and dropped: cuBLAS grouped GEMMs for the experts (faster in isolation, slower inside the engine: 770 vs 790).

## Before and after

| | engine 0.1.12 (`--prefill 2048`) | engine 0.1.13 (`--prefill auto`) |
|---|---:|---:|
| Q2_0, 32K prompt | 572 | 1,290 |
| IQ3_S, 32K prompt | 383 | 1,208 |
| Server, Q2_0, 128K context, 999-token prompt | 353 | 438 |
| Server, 6,927-token prompt | 529 | 1,077 |
| Server, 28,584-token prompt | 584 | 1,249 |

Output speed is unchanged (the decode path is the same). Short prompts (a few hundred tokens) go through the verify
windows as before.

## Quality

- "Bit-identical" steps were checked with the GDN state hash after the 32K prompt (all 36 recurrent layers).
- The steps that change rounding (the batched PLE block, MMQ's 8-bit activations) were compared teacher-forced: the
  32K prompt is read by the prompt path up to its last 256 tokens, those go through the token path with their logits
  dumped. At 32K every rounding change moves the predictions about this much - even a different chunk size of the
  unchanged engine - so that is the yardstick:

| Against the 8192-chunk FP16 path | Same top-1 | Mean KL | Perplexity (reference 9.21) |
|---|---:|---:|---:|
| Chunk 6144, same code (the yardstick) | 89.8% | 0.33 | 9.18 |
| MMQ experts | 85.9% | 0.38 | 8.63 |

- Needles (the answer hidden at a given depth): 5 of 5 at 1K, 32K, 128K (two depths) and 262K, Q2_0, KV streaming.
- A rerun of the same configuration is bit-identical.
