# Layer split: the features that were off across GPUs (2026-09-29)

**Setup:**
- **Rig:** RIBPC. RTX 5080 16 GB (CUDA0) + RTX 3090 24 GB; RTX 2080 Ti 22 GB as a helper card. Ryzen 9 9950X3D,
  62 GB RAM, Windows 11.
- **Model:** the Coder (IQ1_M native pack).
- **Build:** branch `mg/limits`, sm_75 + sm_86 + sm_120.
- **Correctness runs:** `rib_norepeat.py`, with a fixed cache, no swaps, no PCIe share and no suffix drafts where two
  runs are compared byte for byte.
- **Speed runs:** `rib_pp.py`.
- Raw results are in `data/`.

## Correctness

| Check | Result |
|---|---|
| One GPU, this branch vs the 0.1.21-equivalent run (fixed cache) | 10/10 byte-identical |
| Speed projection (control vector), one GPU vs the split on the same GPU (K=24) | 10/10 byte-identical |
| KV streaming (128K context, 32K resident), one GPU vs the split on the same GPU | 10/10 byte-identical |
| Images, one GPU vs the split on the same GPU (three questions about a test picture) | identical answers |
| Speed projection / KV streaming / plain, 5080 + 3090 (K=24): no-repeat under presence 100, the 144 sanity case | all pass, 0 repeats |
| The vector acts across GPUs: 5080 + 3090 with vs without the vector | 8 of 10 answers differ (as on one GPU: 7 of 10) |
| Images, 5080 + 3090 (K=24): "What color is the circle?" / "What number?" / describe | "Red" / "42" / "A red circle, the number 42, and a blue square are displayed on a white background" |
| Mid-prompt checkpoint (every 16,384 tokens): a 19.5K prompt, then another sharing its first 18,600 tokens | one GPU and 5080 + 3090: reused 16,384, answer right |
| 5080 + 3090 (K=30) + a 2080 Ti helper holding 3,000 experts | all pass |
| `--mmap-experts` with the Coder (a native pack) | a clear start-up error: it needs a canonical pack's `experts.bin` (with or without a split) |

Across GPUs the text is not byte-identical to one GPU: experts that one card leaves to the CPU run on a GPU in the
split, and the two round differently.

## Speed

| Run | Prompt 16K / 28K / 60K tok/s | Decode story / code tok/s | Hits |
|---|---|---|---|
| KV streaming (128K), 5080 alone | 1,487 / 1,801 / 1,837 | 66.6 / 74.0 | 59-71% |
| KV streaming (128K), 5080 + 3090 auto | 1,685 / 1,957 / 1,968 | 75.8 / 97.0 | 96-97% |
| Speed projection, 5080 + 3090 auto | 1,813 / 2,053 / - | 77.0 / 103.6 | 96-97% |
| 5080 + 3090 (K=30), no helper | - | 62.5 / 85.2 | |
| 5080 + 3090 (K=30) + 2080 Ti helper | - | 43.0 / 63.7 | |

**The helper card makes it slower,** as it did without a split (bench/results/2026-09-29-layer-split). Its per-layer
round trip costs more than the CPU pool needs for the few experts it would take over. It stays supported: the
helpers take the visible GPUs no stage runs on and hold only experts no stage's cache holds. But it isn't
recommended.

## What changed

- **Per-device tables:** the image-position (mrope) table and the control vector's tables are now kept per device;
  every stage's rope and steering kernels read their own device's copy. `cvec_set_enabled` switches the vector on
  every device.
- **KV streaming:** each stage's session streams its own KV host copy (pinned, portable). The split no longer forces
  it off.
- **Mid-prompt checkpoints:** `Prefill::on_stage_chunk` lets each stage save its part of a checkpoint when it has
  read the chunk. The last stage assembles the parts: the earlier ones are a chunk ahead by then.
- **Helpers (`--expert-cache-remote`) with a split:** they take the visible devices after the stages, skip every
  pair a stage's cache holds, and preflight and open on those devices.
- **Explicit `--expert-cache` with a split:** it is capped so CUDA0 keeps room for the prompt path's own buffers.
  Without the cap, the first prompt failed with "device buffers ... do not fit".
- **`--mmap-experts`:** refused at start for a native pack, with the reason.
