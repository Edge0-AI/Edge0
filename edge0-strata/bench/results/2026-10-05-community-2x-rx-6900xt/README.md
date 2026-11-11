# Community benchmark: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), Ryzen 5 5600X, 128 GB RAM

Measured on 2026-10-05 on a Linux desktop. This tests the original Qwen3.8-Flash-Next (GSQ-RCO IQ3_S) with a
131,072-token context in the three ways two cards can be used - one card alone, a layer split across both, and one card
with the other as an expert helper (`--remote-expert-opt`) - on stock engine 0.1.39 and on 0.1.39 with the three
gfx103x pull requests #835, #849 and #854 applied.

On stock 0.1.39, one card reads a prompt at about **460 tok/s** and decodes at 40-43 tok/s; with the three PRs the same
card reads at **868 / 1,100 / 1,023 tok/s** (4K / 32K / 128K). The layer split with the PRs reads a 128,000-token prompt
at **1,823 tok/s** (70 s to the first token instead of 279 s on one stock card), and the helper mode with #854 decodes at
**60-65 tok/s**. All 42 recall checks (six needles per configuration) were found. The requests are synthetic
code-explanation requests with greedy decoding and a 256-token output cap, and every configuration ran with the engine's
adaptive expert swaps turned off (`--adapt-every 100000`) because of #884; they do not establish answer quality or
performance on other workloads.

## Hardware and software

- **GPUs:** 2x AMD Radeon RX 6900 XT 16 GB (Navi 21, gfx1030, wave32, no matrix cores), reference cards at the driver's
  default 245 W power cap, no clock changes, no display attached. Both on CPU PCIe 4.0 lanes of an X570 board. The
  engine's own probe measured 14.1 GB/s host to device on each card (13.6 in one run), and 28.1 GB/s on both at once
  in an earlier test, which is what the x8/x8 split of the CPU's 16 lanes gives; the cards' sysfs `current_link_width`
  reports x16 for each, which that bandwidth contradicts. No P2P between the cards is used by any of the modes below.
- **CPU and RAM:** AMD Ryzen 5 5600X (6 cores / 12 threads, AVX2, no AVX-512; the engine ran 5 expert-pool workers plus
  its host thread). 128 GB DDR4-3200 (4 DIMMs). 8 GiB swap file.
- **Storage:** the model files and the pack on one NVMe SSD (Samsung 970 EVO 500 GB), also the system disk. The PLE
  table is read from the GGUF shard on it (`--ple-io direct`, the default).
- **OS and runtime:** Ubuntu 26.04, kernel 7.0.0-38-generic with its own amdgpu driver (no DKMS), ROCm 10.0.0 (HIP
  7.15.26333, rocBLAS 5.6.0). hipBLASLt ships no gfx1030 kernels, so the plain hipBLAS/rocBLAS path runs.
- **Source, stock:** [Niko1221/Strata](https://github.com/Niko1221/Strata) `main` at `6f32ec0` (engine 0.1.39).
- **Source, "PRs":** `6f32ec0` with #835 (`f8a6f4a`: the prompt path's 16-bit GEMMs FP16 in and out on gfx103x),
  #849 (`a649e42`: PR #540's attention kernel on gfx103x with 8 cells per step and DPP lane exchanges, bit-exact) and
  #854 (`ffca1b0`: a helper GPU's experts kept out of the PCIe share) merged locally. The PR branches since gained
  docs and `bench/results` commits only; the engine source is the same.
- **Build:** both trees built by hand with cmake and ROCm's clang: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_PREFILL_MMQ=ON` (the options setup uses for AMD), llama.cpp at the pinned
  `3cf0325`. Each configuration's `BUILD.txt` has the tree, the commit and the binary's SHA-256.
- **Other load:** nothing else used the GPUs during the runs (telemetry, below). The machine's other services (a
  llama-swap router that was idle, a home-automation stack) kept running on the CPU.

## Model and configuration

**Model:** [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
IQ3_S - the same two files setup downloads for its IQ3_S choice, downloaded on 2026-09-30 and packed by hand on
2026-10-04 with the repository's `tools/iq_pack.py` (no `--compat-bf16`). The SHA-256 hashes match the repository's
`.sha256` files:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

The MTP draft layer was prepared with the repository's tools. The expert profile is the repository's own
`data/expert-profile.bin` (unchanged). No vision encoder, no calibration, no speed projection.

**Configurations** (each one's `config.json` is in its folder; `/m` is the model directory, `/work/stock` and
`/work/prs` the two trees). Common to all: context 131,072, `--kv int8 --kv-resident 32768` (32,768 K/V cells per
attention layer in VRAM, the rest streamed from pinned RAM), `--expert-cache auto`, `--prefill auto` (8,192-token
chunks, a 384-slot streaming ring), MTP with `--spec 4 --spec-min-p 0.5`, suffix drafting on (the default), and
`--adapt-every 100000`.

| name | cards | engine | notes |
| --- | --- | --- | --- |
| `single-stock` | one (PCI 0f) | stock | expert cache 4,861 slots (9.26 GiB); the prompt path borrows 2,249 of them |
| `single-prs` | one | PRs | the same |
| `split-stock` | both, `layer_split: auto` | stock | layers 0-24 on the first card, 25-47 with the head and the draft on the second (the split's own estimate: 11,176 of the 24,576 (layer, expert) pairs in VRAM, ~97.7% of the profile's routed mass); caches 6,176 slots (11.12 GiB) + 4,829 (9.72 GiB); the prompt path borrows 2,249 + 2,115 of them |
| `split-prs` | both | PRs | the same |
| `helper-stock` | one + the other as helper (`--expert-cache-device1 auto --remote-expert-opt`) | stock | primary 4,861 slots + helper 8,163 (15.46 GiB); the probed PCIe share, `pcie_frac` 0.39 |
| `helper-stock-pcie0` | the same | stock | `--pcie-frac 0`, the workaround for the share bug #854 fixes |
| `helper-prs` | the same | PRs | the probed share 0.39, with #854 |

**Why `--adapt-every 100000`:** on this machine the adaptive expert swaps, the MMQ prompt path and SDMA copies
together hang the engine on the first verify window after a long prompt once about seven requests have been served
(#884; stock hung in 5 of 5 runs of a sequence like this benchmark's). Turning the swaps off avoids it (0 hangs in
7 runs there, and none in the 7 configurations here). It costs decode: with the swaps on, the helper mode on this
machine decodes 66-68 tok/s on ~5K prompts instead of the 60-65 below, and one card 44-48 instead of 40-43. So the
decode numbers here are a few percent under what the engine does when it does not hang.

## Method

[benchmark.py](benchmark.py) is the script of the
[RTX 5090 report](../2026-09-30-community-rtx-5090/README.md), unchanged. It builds deterministic synthetic Python
filler, puts a different nonce near the start of each request, and counts the rendered chat prompt with Strata's
tokenizer. [monitor.py](monitor.py) is the AMD form of that report's monitor: it samples RAM, swap and both cards'
VRAM, GTT, busy percentage and power from sysfs once per second.

Every configuration started its own server (`serve/server.py --engine strata --config <config.json> --port 8089`
from its tree), and ran, in this order:

```bash
python3 benchmark.py --root /work/<tree> --pack /m/packs/iq3_s --url http://127.0.0.1:8089 --out <folder> --targets 4096,32768,128000 --runs 3
python3 /work/<tree>/tools/needle_bench.py --url http://127.0.0.1:8089 --lengths 32k,128k --depths 10,50,90 --out <folder>/needles.json
```

- One short warm-up request is excluded.
- Three runs at each length ran serially, in increasing-length order, on the same loaded engine.
- All 63 speed requests read their whole prompt: **zero reused tokens**; every one generated 256 tokens and stopped
  at the output cap (`results.json`).
- The expert cache was filled from the profile at start and kept between requests. Loading time is excluded.
- TTFT is streaming, from just before the HTTP request to the first nonempty text delta, over loopback.
- Prompt throughput is freshly read tokens / `prompt_ms`; decode throughput is `engine_generated / decode_ms`, both
  from the engine's own accounting (`/metrics`).
- `reasoning_effort: none`, temperature 0.
- Each folder has the engine's whole log for its session (`engine.log`), the benchmark's output, `results.json`
  (request hashes and every output text), `summary.json`, `needles.json`, the server's status before
  and after, and `BUILD.txt`.

The seven servers ran one after the other between 02:58 and 06:02 local time, with one interruption: after
`split-stock` the sequence was paused for an unrelated test and resumed 10 minutes later. Each configuration is a fresh
engine, so the pause changes nothing in them.

## Results

Each cell is the median **[minimum-maximum]** of three runs. No speed request failed, was cancelled or stalled.

### One card (PCI 0f)

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 4,096 | 457.0 [451.1-458.6] | 40.2 [38.2-41.3] | 9.00 [8.97-9.12] | 15.46 [15.14-15.67] |
| stock | 32,768 | 473.6 [453.7-473.7] | 43.4 [41.9-44.2] | 69.37 [69.25-72.30] | 75.13 [75.13-78.37] |
| stock | 128,000 | 458.6 [458.5-458.9] | 41.4 [38.3-42.2] | 279.29 [279.11-279.35] | 285.51 [285.33-285.77] |
| PRs | 4,096 | **868.0** [825.2-869.7] | 42.0 [38.9-42.9] | 4.76 [4.75-5.00] | 10.82 [10.69-11.56] |
| PRs | 32,768 | **1,099.5** [1,099.4-1,102.3] | 42.9 [40.3-44.3] | 29.88 [29.80-29.88] | 35.82 [35.55-36.20] |
| PRs | 128,000 | **1,023.4** [1,023.0-1,024.8] | 39.6 [37.8-39.7] | 125.27 [125.10-125.33] | 131.76 [131.68-131.85] |

### Layer split across both cards

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 4,096 | 445.6 [438.6-446.7] | 58.9 [58.2-59.3] | 9.23 [9.20-9.37] | 13.60 [13.53-13.67] |
| stock | 32,768 | 722.7 [721.1-724.1] | 60.6 [60.0-60.9] | 45.41 [45.32-45.51] | 49.61 [49.57-49.69] |
| stock | 128,000 | 820.8 [820.7-821.1] | 58.2 [52.5-58.7] | 156.14 [156.09-156.16] | 160.47 [160.46-161.01] |
| PRs | 4,096 | **852.5** [806.3-853.6] | 59.3 [57.2-60.5] | 4.84 [4.83-5.11] | 9.29 [9.05-9.40] |
| PRs | 32,768 | **1,616.1** [1,614.2-1,620.1] | 59.0 [56.7-62.4] | 20.35 [20.30-20.37] | 24.66 [24.38-24.86] |
| PRs | 128,000 | **1,823.1** [1,821.0-1,824.9] | 57.4 [56.7-58.6] | 70.40 [70.33-70.48] | 74.82 [74.76-74.89] |

### One card with the other as expert helper

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock, probed share 0.39 | 4,096 | 457.3 [449.1-458.2] | 42.3 [41.8-42.4] | 8.99 [8.98-9.16] | 15.02 [14.99-15.26] |
| stock, probed share 0.39 | 32,768 | 473.0 [470.0-473.7] | 40.5 [40.0-41.4] | 69.35 [69.25-69.79] | 75.73 [75.41-76.10] |
| stock, probed share 0.39 | 128,000 | 458.4 [458.2-458.6] | 38.0 [37.8-41.1] | 279.42 [279.33-279.56] | 286.08 [285.62-286.27] |
| stock, `--pcie-frac 0` | 4,096 | 459.5 [452.7-459.8] | 62.8 [58.4-64.4] | 8.95 [8.94-9.08] | 12.99 [12.90-13.44] |
| stock, `--pcie-frac 0` | 32,768 | 473.3 [473.0-473.6] | 62.0 [59.2-64.1] | 69.31 [69.26-69.34] | 73.45 [73.23-73.61] |
| stock, `--pcie-frac 0` | 128,000 | 458.1 [458.0-458.4] | 56.3 [53.6-59.8] | 279.62 [279.43-279.64] | 284.16 [283.88-284.18] |
| PRs, probed share 0.39 | 4,096 | **875.4** [826.9-875.9] | **64.5** [62.7-68.9] | 4.71 [4.71-4.98] | 8.66 [8.40-9.05] |
| PRs, probed share 0.39 | 32,768 | **1,101.1** [1,100.9-1,101.8] | **62.5** [62.2-67.6] | 29.83 [29.81-29.84] | 33.90 [33.57-33.93] |
| PRs, probed share 0.39 | 128,000 | **1,024.0** [1,024.0-1,024.4] | **60.0** [58.6-62.8] | 125.19 [125.14-125.20] | 129.43 [129.25-129.49] |

What the tables show:

- **#835 and #849** (the prompt-path PRs) read prompts 1.9x faster at 4K and 2.2-2.3x at 32K and 128K, on one card and
  on the split alike; decode is unchanged (it does not use those kernels). The 128K first-token time on one card goes
  from 279 s to 125 s, on the split from 156 s to 70 s.
- **The layer split** reads long prompts faster than one card (each card runs its own layers) but 4K prompts
  slightly slower, 446 against 457 tok/s on stock; a single 4,096-token chunk likely does not amortise the hand-over
  between the cards, which was not measured separately. Its decode, 57-61 tok/s, is the same with and without the
  PRs.
- **The helper mode on stock** decodes no faster than one card (38-42 tok/s): the expert plan's PCIe share takes
  experts the helper already holds and the primary computes them (#854). `--pcie-frac 0` is the workaround (56-63) and
  #854 the fix (60-65, with the probed share left on). The helper mode's prompt read is one card's, since only the
  primary reads prompts; the helper's VRAM is spent on decode.
- Run-to-run spread is small: the three prompt readings at 32K and 128K are within 0.3% of each other in every
  configuration; decode varies more (up to 10%) with draft acceptance, although the text is greedy.
- The decode expert-cache hit rate per request (each `engine.log`) was 32-58% on one card, 59-86% on the split and
  40-87% in the helper modes, lowest on the first requests: these synthetic prompts route to many experts, and the
  one-card cache holds 4,861 of the 24,576. The CPU pool (and, in the helper mode, the second card) computes the rest,
  which is where the decode time goes on one card.

**Memory and power**, from each configuration's `telemetry.jsonl` (one-second samples from the server's start through
the recall checks; the raw files, 4 MB in all, are not kept in this repository):

- VRAM peak: 15.9 GiB on the primary card in every configuration (`mem_info_vram_used`, the whole card); 15.9-16.0 GiB
  on the second card in the split and helper modes. In `single-stock`, the idle second card showed 9.5 GiB in use for
  18 s while the engine loaded (0% busy, 5 W), then 16 MiB for the rest of the run; `single-prs` did not.
- RAM: `MemAvailable` never below **65.7 GiB** of 121.5; swap: no growth (0.36 GiB from earlier use, 0.93 in the first
  run).
- Power: the loaded card's `power1_average` peaked at 250-254 W (the cap is 245 W; the average can overshoot it briefly);
  the helper card at 66-74 W, the second split card at 251-254 W.

These are sampled values; brief peaks between samples can be missed.

## Recall and limitations

The repository's unchanged `tools/needle_bench.py` found **all six needles** in each of the seven configurations, at
depths 10%, 50% and 90% at both lengths ([needles.json](single-stock/needles.json) and the others): the `32k` prompts
were 32,171-32,172 tokens and the `128k` prompts 125,449-125,451 tokens, all read in full (no reused prefix).

**Limits of this report:**

- One machine, one quantization (IQ3_S, packed by hand from setup's files), one context size and a small synthetic
  workload. Long output, sampled decoding, thinking, coding-task correctness, vision, tool use, concurrency and a
  sustained thermal run were not part of it.
- The adaptive expert swaps were off in every configuration (#884, above), so decode is a few percent under the
  engine's best on this machine.
- The "PRs" numbers are for pull requests that were open when this was measured; #835 changes the prompt path's
  arithmetic (FP16 GEMM outputs; its distribution check is in
  [bench/results/2026-10-04-rdna2-fp16-prompt](../2026-10-04-rdna2-fp16-prompt/README.md)), #849 and #854 do not
  change any value.
- For context, on the same machine and model, llama.cpp (3cf0325, ROCm, with its own expert-cache patches) measured
  26 tok/s decode and 274 tok/s on a 37K prompt on 2026-10-04. That is a different engine and workload, not a
  controlled comparison.
