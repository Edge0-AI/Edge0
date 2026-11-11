# 2026-09-27: expert cache on vs off (issue #23)

Question: the engine used to warn at start that "the GPU hit path is NOT CORRECT ... any OUTPUT from it is not". Does
the expert cache make the output worse?

Method: teacher-forced. `strata --dump-logits` runs a text one token at a time and writes the logits at every
position, so both runs see the same text and each position's next-token prediction can be compared:
- cache off: no `--expert-cache`, every routed expert computed by the CPU pool;
- cache on: `--expert-profile pack/profile-decode-8k.bin --expert-cache auto`, 5,130 of the most-used experts on the GPU.

Q2_0, RTX 5070, the first 512 or 1,024 tokens of three frozen prompts.

| Text | Same top-1 | Median KL | Perplexity off | Perplexity on | On - off (nats per token) |
|---|---:|---:|---:|---:|---:|
| Code agent (512) | 97.7% | 3.0e-04 | 2.672 | 2.630 | -0.016 +- 0.019 |
| Long document (1,024) | 96.2% | 4.5e-03 | 29.844 | 29.843 | -0.000 +- 0.005 |
| Chat (1,024) | 95.0% | 4.4e-03 | 18.380 | 18.301 | -0.004 +- 0.006 |
| **All 2,557 tokens** | | | | | **-0.005 +- 0.005** |

Reading:
- The cache changes which of two nearly equal candidates wins at 2-5% of positions. Most of the flips have a top-2
  margin under 0.5 logits (`logits-compare.txt`).
- Neither side is more accurate: the perplexity of the actual text is the same, within 1 standard error.
- A GPU expert and the CPU compute the same quantized expert with a different floating-point order. Both are an
  equally good approximation of the model.

The start-up warning dates from round 328, when the GPU hit path had a real fault (divergence from token 0). That fault
was fixed long ago and is covered by `expert_parity`, `native_expert_parity` and the grouped kernels' tests. Since
engine 0.1.8 the warning is replaced by one accurate line.
