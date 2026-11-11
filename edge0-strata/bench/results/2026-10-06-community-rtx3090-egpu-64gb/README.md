# Community benchmark — RTX 3090 in Thunderbolt eGPU, 64 GB RAM — 2026-10-06

Two quantizations of Qwen3.8-Flash-Next on the same machine, same day, same settings
(only the quantization changes). All prompt tokens are freshly processed: every request
uses a unique nonce so no prefix-cache reuse occurs (reused tokens = 0 everywhere).

## Hardware

- GPU: NVIDIA GeForce RTX 3090, 24 GB, **in a Thunderbolt 3 eGPU enclosure** (single-GPU
  configuration; the internal RTX 3070 Laptop 8 GB is present in the system but unused by
  Strata — a multi-GPU 3090+3070 split was also measured and rejected, see Notes)
- CPU: 11th Gen Intel Core i7-11800H (8C/16T, AVX-512)
- RAM: 64 GB DDR4
- Storage: NVMe SSD
- PCIe link to GPU: Thunderbolt 3 (~PCIe x4 equivalent)
- Other workloads during runs: idle desktop only

## Software

- OS: Windows 11 Pro 26H2 (build 26300.9457)
- Strata: commit `1678de333d0e0711bc414ad992b640e1a37dd814` (2026-10-02, shallow clone)
- Engine: prebuilt `strata.exe` 0.1.34 (sm_86, CUDA 13.0), NVIDIA driver 617.14 WHQL

## Model

- Qwen3.8-Flash-Next, Strata GSQ-RCO packs
- Quantizations tested: **IQ2_XS** and **IQ3_XXS**
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf`, same pattern for IQ3_XXS)
- Vision encoder loaded (mmproj BF16) but not used in these tests
- MTP draft layer: on

## Settings (identical for both configurations)

- Launch: `START-HERE.bat --setup --yes --family qwen --model <quant> --gpu 1 --vision yes
  --context 262144` (host 0.0.0.0, API key set — key not included anywhere in this PR)
- Context: 262144 (native), KV int8, KV streaming on (KV cache in RAM)
- Calibration: `START-HERE.bat --calibrate` run per model → both tuned to
  `--pcie-frac 0.00, --spec-min-p 0.70`, 7 CPU workers
- Sampling: temperature 0

## Workload

- Script: `bench.py` (this folder), 3 sequential runs per configuration
- Prefill tests: generated filler text with a per-request unique nonce (defeats prefix
  cache), three target sizes; answer capped at 128 tokens ("Reply with exactly: OK")
- TTFT: streaming request, first token is a reasoning token (hybrid thinking model),
  keep-alives and empty deltas skipped; ~500-token cap
- Parallel: 3 concurrent short requests (the server serves them from one queue)
- Needle (correctness): a secret code embedded at the head of ~25.7K fresh tokens,
  asked back at the end of the context; auto-verified
- Warm-up: none dedicated; each configuration's run 1 starts ~2 minutes after model
  load. Per-run values are in the JSON files; ranges below include run 1
- Failures: none (all requests completed, no paging, no OOM)

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| IQ2_XS | 1748 | 0 | 65 | 3 | 188.3 (188.2–188.8) | 112.2 (106.2–116.9) | not measured |
| IQ2_XS | 6488 | 0 | 72 | 3 | 560.2 (560.0–560.3) | 101.1 (97.0–114.8) | not measured |
| IQ2_XS | 26990 | 0 | 63 | 3 | 618.1 (617.6–618.3) | 111.5 (105.1–115.2) | not measured |
| IQ2_XS (TTFT, short prompt) | ~25 | 0 | 419 | 3 | not measured | 81.1 streamed (79.2–81.8) | 2.171 (2.069–2.811) |
| IQ3_XXS | 1748 | 0 | 68 | 3 | 128.0 (127.9–128.3) | 60.2 (54.8–70.7) | not measured |
| IQ3_XXS | 6490 | 0 | 69 | 3 | 421.4 (421.4–421.7) | 74.2 (74.0–76.7) | not measured |
| IQ3_XXS | 26991 | 0 | 42 | 3 | 467.6 (467.4–467.7) | 81.3 (59.2–89.2) | not measured |
| IQ3_XXS (TTFT, short prompt) | ~25 | 0 | 211 | 3 | not measured | 60.3 streamed (58.5–65.2) | 3.587 (3.533–4.612) |

Decode tok/s comes from the server-reported per-request timings (predicted tokens ÷
generation time), never from generated tokens ÷ total request time. Decode is measured
on short generations (42–72 tokens for prefill rows; the TTFT row streams ~200–420
tokens including reasoning).

Correctness (needle): found 6/6 (3/3 per configuration) at ~25.7K fresh tokens each.

Parallel (informational, not in the table): 3 concurrent clients are served from a
single queue; total wall time for 3 × ~200-token answers: 13.5–14.2 s (IQ2_XS),
19.7–23.2 s (IQ3_XXS).

Memory: with IQ2_XS resident, ~39 GB RAM used by the model, ~23 GB free; with IQ3_XXS,
~47 GB / ~16 GB free. VRAM used ~23.3 GB of 24 GB (expert cache ~17 GiB, IQ2_XS).

## Notes

- A 3090 (24 GB) + 3070 Laptop (8 GB, internal) layer-split configuration was also
  measured on this machine (single run, not part of this table): decode dropped to
  ~66 tok/s, prefill to ~128–149 tok/s and the setup reduced the usable context to
  32K — the much slower second card hurts here, consistent with the MULTI_GPU.md
  guidance about much slower extra cards.
- The eGPU Thunderbolt link does not bottleneck Strata's architecture: calibration
  picked PCIe share 0.00 (no expert prefetch over the link) on both quants.
- Raw per-run data: `runs/*.json`. The script reads the API key from the
  `STRATA_API_KEY` environment variable; no credentials or machine-specific paths are
  included in this PR.
