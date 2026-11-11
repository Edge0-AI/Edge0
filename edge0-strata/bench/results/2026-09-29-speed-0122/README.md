# Speed with engine 0.1.22

RTX 5070 12 GB (PCIe 5.0 x16), Ryzen 5 7600, 64 GB DDR5-5200, Windows 10. The ready-made 0.1.22 engine, one-shot runs
with what setup writes: `--prefill auto`, the shipped expert profile, `--expert-cache auto`, 8-bit KV above 4K, KV
streaming (`--kv-resident 32768`) from 64K, MTP (`--spec 4 --spec-min-p 0.5`), greedy, 256 generated tokens, images
off. The same code-agent prompts as `2026-09-28-speed-0114`. Every run: [`matrix.json`](matrix.json).

## Prompt (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 519 | 1,226 | 1,844 | 1,836 | 1,682 | 1,304 |
| **IQ2_XS** | 524 | 1,196 | 1,799 | 1,611 | 1,495 | 1,181 |
| **IQ3_XXS** | 472 | 974 | 1,555 | 1,449 | 1,386 | - |
| **IQ3_S** | 419 | 893 | 1,499 | 1,285 | 1,245 | - |
| **Coder** | 660 | 1,522 | 1,871 | 1,938 | 1,779 | - |

## Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 84.7 | 88.2 | 77.4 | 71.0 | 63.1 | 61.1 |
| **IQ2_XS** | 77.2 | 74.6 | 67.5 | 63.5 | 59.1 | 54.7 |
| **IQ3_XXS** | 65.1 | 62.3 | 49.8 | 53.6 | 47.0 | - |
| **IQ3_S** | 50.5 | 44.6 | 49.0 | 48.8 | 41.3 | - |
| **Coder** | 56.9 | 53.6 | 54.9 | 48.5 | 42.8 | - |

- IQ2_XS was measured with Swift 1.5's IQ2_XS (the same speed as the original's); its 262K run had images on.
- IQ3_XXS and IQ3_S at 262K do not fit 64 GB of RAM; the Coder at 262K was not run again (0.1.14: 1,034 tokens/s).
- Against 0.1.14: prompts 1.2-1.5x faster at 32K-128K (the 0.1.22 prompt path: tensor-core attention, bit-identical
  batched kernels, faster PLE reads). Output speed is the engine's same decode; one run per cell moves a few percent
  with the text (speculative drafts), so compare it across many runs, not one.
