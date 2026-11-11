# Speed with engine 0.1.26

RTX 5070 12 GB (PCIe 5.0 x16), Ryzen 5 7600, 64 GB DDR5-5200, Windows 10. The ready-made 0.1.26 engine, one-shot runs
with what setup writes: `--prefill auto`, the shipped expert profile, `--expert-cache auto`, 8-bit KV above 4K, KV
streaming (`--kv-resident 32768`) from 64K, MTP (`--spec 4 --spec-min-p 0.5`), greedy, 256 generated tokens, images
off, no timing marks. The same code-agent prompts as `2026-09-29-speed-0122`. Every run: [`matrix.json`](matrix.json).

## Prompt (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 536 | 1,299 | 2,171 | 2,126 | 2,107 |
| **IQ2_XS** | 534 | 1,256 | 2,092 | 1,754 | 1,752 |
| **IQ3_XXS** | 482 | 1,007 | 1,745 | 1,609 | 1,602 |
| **IQ3_S** | 427 | 913 | 1,624 | 1,640 | 1,443 |
| **Coder** | 656 | 1,583 | 2,177 | 2,236 | 2,208 |

## Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 87.3 | 93.0 | 81.8 | 76.2 | 73.7 |
| **IQ2_XS** | 79.6 | 78.6 | 76.3 | 63.7 | 62.7 |
| **IQ3_XXS** | 61.9 | 61.6 | 58.5 | 57.2 | 49.0 |
| **IQ3_S** | 52.4 | 53.3 | 48.3 | 46.3 | 45.5 |
| **Coder** | 58.9 | 55.1 | 54.9 | 53.2 | 43.0 |

- Against 0.1.22 (`2026-09-29-speed-0122`), prompts at 32K-128K are 8-28% faster: Q2_0 1,844 -> 2,171 at 32K and
  1,682 -> 2,107 at 128K, the Coder 1,779 -> 2,208 at 128K. The gains come from:
  - 0.1.24's QSA selection on tensor cores;
  - 0.1.25's grouping tables in mapped memory and fused norms;
  - 0.1.26's batched draft-layer pass.
- 262K was not run again (0.1.22: Q2_0 1,304, IQ2_XS 1,181 tokens/s).
- Output speed is the same decode as before. One run per cell moves several percent with the text (the share of
  accepted drafts, `spec_accept` in `matrix.json`). For example, the Coder's 128K answer accepted 55% of its drafts
  and its 64K answer 71%.
