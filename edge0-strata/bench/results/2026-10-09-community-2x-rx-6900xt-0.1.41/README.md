# Community benchmark: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), Ryzen 5 5600X, 128 GB RAM - Strata 0.1.41: one card, layer split, expert helper, and the decode window

Measured on 2026-10-08/09 on the same Linux desktop as the [2026-10-05](../2026-10-05-community-2x-rx-6900xt/README.md)
and [2026-10-06 (0.1.40.1)](../2026-10-06-community-2x-rx-6900xt-0.1.40.1/README.md) reports. It covers the three ways
the two cards can be used - one card alone, the layer split (this machine's production mode) and one card with the other
as an expert helper - and asks three questions:

1. What does **0.1.41 as released** do on this card family against 0.1.40.1 - stock, and with the two gfx103x switches
   it ships (`STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1`)?
2. What do the **three open gfx103x pull requests** (#1149, #1151, #1167), plus a per-device rocBLAS solution cache and
   a modified llama.cpp ggml for the expert MMQ, add on top of 0.1.41?
3. Can the **decode** be moved by configuration alone? The 0.1.40.1 report left decode at 62-75 tok/s in every arm.
   Here a shorter MTP draft with a higher acceptance threshold **together with** `--pipeline-windows 2` decodes
   8-16% faster; each of the two alone had been measured as neutral or worse.

The short answer: stock 0.1.41 reads prompts 4-6% faster than stock 0.1.40.1 on the split (464 / 767 / 869 tok/s at
4K / 32K / 128K), the two switches 5-12% faster (832 / 1,619 / 1,823), and the pull requests land where they did on
0.1.40.1 (968 / 1,714 / 2,029 - the 128K prompt in 63 s to the first token). Decode is unchanged by the engine version;
`--spec 3 --spec-min-p 0.7 --pipeline-windows 2` takes it from 64.7 / 68.8 / 68.3 to 77.2 / 82.3 / 75.7 tok/s (4K / 32K / 128K) on the same binary. No arm stalled.

The adaptive expert swaps were on (the default) in every arm. GFXOFF stayed at the kernel's default; instead of the
debugfs knob used in the 0.1.40.1 report, the two-card arms that read 128K prompts ran with the engine's own
`STRATA_HIP_ADAPT_KERNEL_COPY=1` (in 0.1.41; the adaptive swaps copy with a kernel instead of SDMA, which on this
machine removes the #884 stall at no measurable cost - the `split-stock` and `split-stock-kc` rows below are the A/B).

## Hardware and software

Unchanged from the 0.1.40.1 report except the engine:

- **GPUs:** 2x AMD Radeon RX 6900 XT 16 GB (Navi 21, gfx1030, wave32, no matrix cores), reference cards, 245 W power
  cap, no clock changes, no display attached. Both on CPU PCIe 4.0 lanes of an X570 board, x8 each (the engine's probe:
  14.1 GB/s host to device). No P2P between the cards.
- **CPU and RAM:** Ryzen 5 5600X (6 cores / 12 threads, AVX2), 128 GB DDR4-3200 (4 DIMMs), 8 GiB swap file.
- **Storage:** model, pack and PLE table on one NVMe SSD (Samsung 970 EVO 500 GB). `--ple-io ram` (the PLE table is
  read into RAM at start, `memlock` unlimited); the 0.1.40.1 report used the default `direct`.
- **OS and runtime:** Ubuntu 26.04, kernel 7.0.0-38-generic with its own amdgpu driver, ROCm 10.0.0 (HIP 7.15, rocBLAS
  5.6.0.8d1ae90e). hipBLASLt ships no gfx1030 kernels; the plain rocBLAS path runs.
- **Source, stock:** [Niko1221/Strata](https://github.com/Niko1221/Strata) `main` at `fb58e0db` (engine 0.1.41,
  2026-10-08), ggml from llama.cpp at the pinned `3cf03257f` (`STRATA_GGML_DIR` pointed at a checkout of that commit;
  the FetchContent default fetches the same one). Built with
  `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_PREFILL_MMQ=ON`.
- **Source, "PRs":** `fb58e0db` with the open pull requests #1149 (FP16 prompt route: tests and the
  `STRATA_F16_RANGE` counter), #1151 (the QSA block scores on rocBLAS SGEMM, `STRATA_SELECT_SGEMM=1`) and #1167 (a
  rocBLAS solution table for the FP16 prompt GEMMs, `STRATA_ROCBLAS_TUNING`) rebased on top, plus two local commits
  that keep the rocBLAS solutions in a per-device cache file (`STRATA_TUNING_CACHE`; the solution is measured once on
  the card and read back on later starts, so it no longer depends on the start-up heuristic). Same build options,
  plus `-DSTRATA_GGML_DIR` pointing at llama.cpp `159c651f5` with six RDNA2 MMQ patches (wider tiles, 24-bit scale
  multiplies, Q8_K-style activations, VOP3P dot-chain heads; the patches, build flags and their own measurements are in
  [Vop3p/dual-6900xt-flash-next-tuning](https://github.com/Vop3p/dual-6900xt-flash-next-tuning/tree/main/patches/llama.cpp)).
  Strata takes its expert prompt GEMMs from that MMQ; on 0.1.40.2 the ggml change alone was +1.5% at a 32K prompt with
  byte-identical output. The PR commits and the ggml change were not separated again here.
- **Other load:** nothing else used the GPUs (telemetry); an idle llama-swap router, a home-automation stack and the
  LACT fan daemon kept running on the CPU. No builds during the runs.

## Model and configuration

**Model:** [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
IQ3_S, the same files and pack as the two earlier reports (hashes in the 2026-10-05 README); MTP draft layer prepared
with the repository's tools; the repository's `data/expert-profile.bin`. No vision encoder, no calibration, no speed
projection.

**Common to all arms:** context 131,072, `--kv int8 --kv-resident 32768`, `--expert-cache auto`, `--prefill auto`
(8,192-token chunks), MTP `--spec 4 --spec-min-p 0.5` unless the arm says otherwise, `--ple-io ram`,
`--vram-reserve-mib 700`, cards given as `[1, 0]` with `"gpu_order": "as_given"` (new in 0.1.41: without it the engine
reorders layer-split cards by SIMDs x clock), `layer_split: auto`. The split landed at **layers 0-24 on the first card
and 25-47 on the second in every arm** (6,176 + 4,829 expert slots).

| arm | tree | environment | prompt lengths |
| --- | --- | --- | --- |
| `single-stock` | stock | none (one card, PCI 0f; GFXOFF default) | 4K, 32K, 128K |
| `single-sw` | stock | `STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1` | 4K, 32K, 128K |
| `single-prs` | PRs | the production environment (as `split-prs-kc` below, without kernel-copy) | 4K, 32K, 128K |
| `helper-stock` | stock | `STRATA_HIP_ADAPT_KERNEL_COPY=1`; the other card as expert helper (`--expert-cache-device1 auto --remote-expert-opt`, probed PCIe share 0.39) | 4K, 32K, 128K |
| `helper-sw` | stock | the two switches + kernel-copy, helper as above | 4K, 32K, 128K |
| `helper-prs` | PRs | the production environment + kernel-copy, helper as above | 4K, 32K, 128K |
| `split-stock` | stock | none | 4K, 32K (128K left out: at the GFXOFF default a stock split stalled on its first 128K request in the 0.1.40.1 report, #884) |
| `split-stock-kc` | stock | `STRATA_HIP_ADAPT_KERNEL_COPY=1` | 4K, 32K, 128K |
| `split-sw-kc` | stock | `STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1 STRATA_HIP_ADAPT_KERNEL_COPY=1` | 4K, 32K, 128K |
| `split-prs-kc` | PRs | the three above + `STRATA_ROCBLAS_TUNING=/work/prs/tools/hip STRATA_TUNING_CACHE=~/.cache/strata STRATA_SELECT_SGEMM=1 STRATA_SELECT_SGEMM_VERBOSE=1 STRATA_IO_THREADS=128 STRATA_PREFILL_CPU_SHARE=auto` (this machine's production environment) | 4K, 32K, 128K |
| `split-prs-kc-dec` | PRs | as `split-prs-kc`, with `--spec 3 --spec-min-p 0.7 --pipeline-windows 2` | 4K, 32K, 128K |

The engine's log confirms each switch (`engine.log` was kept locally; the lines are quoted in `BUILD.txt`'s folder):
`STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out` once per card, `prefill gemm: rocBLAS tuning
enabled (112 rows, 76 with a solution, gfx1030, mp 40, rocBLAS 5.6.0.8d1ae90e)`, `qsa select: SGEMM reach<=... solution
... from .../gfx1030-mp40-rocblas-5.6.0.8d1ae90e-select.txt`, `STRATA_HIP_ADAPT_KERNEL_COPY=1: adaptive swaps copy with
a kernel, not SDMA (#884)`, and for the last arm the pipeline-windows line.

## Method

[benchmark.py](benchmark.py) is the script of the earlier reports (the RTX 5090 report's, unchanged): deterministic
synthetic Python filler with a different nonce near the start of each request, counted with Strata's tokenizer. Every
arm started its own server (`serve/server.py --engine strata --config <config.json> --port 8089` from its tree) and
ran `benchmark.py --targets 4096,32768,128000 --runs 2` (two runs per length, serial, increasing length, after one
excluded warm-up). All speed requests read their whole prompt (zero reused tokens) and generated 256 tokens to the
output cap. Prompt throughput is freshly read tokens / `prompt_ms`, decode throughput the engine's `decode_tok_s`;
TTFT is streaming over loopback. `reasoning_effort: none`, temperature 0. The recall checks (`needle_bench.py`) were
not repeated: the 0.1.40.1 report found 54 of 54 on this machine and nothing in these arms changes the model's
numerics beyond the FP16 prompt route already covered there.

The eleven servers ran one after the other: the five split arms between 23:47 and 00:13 local time, the one-card and
helper arms between 03:09 and 04:03 (the one-card and helper configurations are the 0.1.40.1 report's, with the paths
of this machine and `--ple-io ram` added to match the split arms; the helper arms ran with the kernel-copy switch
because the debugfs GFXOFF knob the 0.1.40.1 report used needs root); each folder has `config.json`,
`summary.json`, `results.json` (request hashes, every output text, the engine's per-request accounting) and
`BUILD.txt` (tree, binary SHA-256, environment, GFXOFF reading at start). The full engine logs and one-second telemetry
are in the companion repository's `bench/results/e388/`.

## Results

Each cell is the median **[minimum-maximum]** of two runs. No request of the eleven arms failed, was cancelled or
stalled.

### One card (PCI 0f), GFXOFF default

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 4,096 | 472.9 [471.8-474.0] | 45.6 [42.5-48.7] | 8.70 [8.68-8.72] | 14.30 [13.90-14.71] |
| stock | 32,768 | 502.1 [502.0-502.2] | 48.0 [46.6-49.4] | 65.32 [65.30-65.33] | 70.62 [70.46-70.79] |
| stock | 128,000 | 486.3 [486.3-486.3] | 46.6 [45.3-47.9] | 263.29 [263.29-263.30] | 268.76 [268.61-268.92] |
| switches | 4,096 | 849.9 [827.3-872.5] | 49.0 [46.2-51.8] | 4.86 [4.73-4.98] | 10.07 [9.64-10.49] |
| switches | 32,768 | 1,102.4 [1,100.8-1,103.9] | 51.6 [49.0-54.1] | 29.77 [29.73-29.81] | 34.72 [34.43-35.01] |
| switches | 128,000 | 1,026.9 [1,026.6-1,027.2] | 49.3 [45.6-53.0] | 124.74 [124.71-124.78] | 129.93 [129.58-130.29] |
| PRs + ggml | 4,096 | 976.2 [958.7-993.7] | 49.0 [46.7-51.2] | 4.23 [4.15-4.31] | 9.44 [9.12-9.76] |
| PRs + ggml | 32,768 | 1,173.6 [1,171.9-1,175.3] | 50.9 [48.8-53.0] | 27.97 [27.93-28.01] | 32.98 [32.81-33.14] |
| PRs + ggml | 128,000 | 1,146.6 [1,146.2-1,147.0] | 49.8 [49.8-49.9] | 111.73 [111.69-111.77] | 116.83 [116.79-116.87] |

### One card with the other as expert helper, `STRATA_HIP_ADAPT_KERNEL_COPY=1`

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock + kernel-copy | 4,096 | 477.5 [473.7-481.2] | 67.7 [62.8-72.6] | 8.61 [8.54-8.68] | 12.38 [12.04-12.73] |
| stock + kernel-copy | 32,768 | 503.2 [503.1-503.4] | 66.5 [66.4-66.5] | 65.16 [65.14-65.17] | 68.98 [68.97-69.00] |
| stock + kernel-copy | 128,000 | 486.8 [486.7-486.9] | 65.2 [60.5-69.9] | 263.03 [262.97-263.09] | 266.96 [266.73-267.18] |
| switches + kernel-copy | 4,096 | 858.6 [834.6-882.5] | 69.0 [63.8-74.2] | 4.80 [4.67-4.94] | 8.51 [8.09-8.93] |
| switches + kernel-copy | 32,768 | 1,103.5 [1,102.4-1,104.6] | 74.9 [70.6-79.1] | 29.74 [29.71-29.77] | 33.14 [32.92-33.37] |
| switches + kernel-copy | 128,000 | 1,027.9 [1,027.7-1,028.1] | 70.7 [70.5-70.9] | 124.61 [124.59-124.63] | 128.21 [128.20-128.22] |
| PRs + ggml + kernel-copy | 4,096 | 986.3 [964.0-1,008.6] | 72.5 [67.5-77.5] | 4.18 [4.09-4.28] | 7.71 [7.37-8.05] |
| PRs + ggml + kernel-copy | 32,768 | 1,172.5 [1,171.9-1,173.1] | 75.2 [70.5-79.8] | 27.99 [27.97-28.00] | 31.38 [31.16-31.61] |
| PRs + ggml + kernel-copy | 128,000 | 1,147.2 [1,146.7-1,147.7] | 75.1 [74.9-75.3] | 111.66 [111.61-111.71] | 115.05 [115.00-115.09] |

### Layer split across both cards

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 4,096 | 463.9 [460.3-467.6] | 62.6 [62.4-62.7] | 8.86 [8.79-8.93] | 12.93 [12.85-13.00] |
| stock | 32,768 | 766.8 [763.1-770.6] | 65.0 [61.8-68.3] | 42.78 [42.57-42.99] | 46.70 [46.68-46.71] |
| stock + kernel-copy | 4,096 | 461.4 [457.1-465.7] | 61.5 [60.4-62.6] | 8.91 [8.82-8.99] | 13.04 [12.89-13.20] |
| stock + kernel-copy | 32,768 | 767.0 [766.5-767.5] | 65.1 [62.0-68.3] | 42.77 [42.74-42.80] | 46.68 [46.52-46.84] |
| stock + kernel-copy | 128,000 | 869.4 [868.1-870.6] | 60.8 [59.7-61.9] | 147.32 [147.11-147.54] | 151.51 [151.22-151.80] |
| switches + kernel-copy | 4,096 | 831.9 [807.6-856.2] | 69.2 [67.8-70.6] | 4.96 [4.81-5.10] | 8.63 [8.41-8.85] |
| switches + kernel-copy | 32,768 | 1,618.6 [1,617.4-1,619.9] | 74.6 [72.0-77.3] | 20.29 [20.27-20.30] | 23.70 [23.56-23.83] |
| switches + kernel-copy | 128,000 | 1,822.8 [1,822.3-1,823.4] | 67.5 [66.0-68.9] | 70.31 [70.28-70.33] | 74.08 [73.98-74.19] |
| PRs + ggml | 4,096 | 968.3 [945.6-991.1] | 64.7 [62.0-67.4] | 4.26 [4.16-4.36] | 8.20 [7.94-8.46] |
| PRs + ggml | 32,768 | 1,714.4 [1,712.5-1,716.2] | 68.8 [68.5-69.1] | 19.15 [19.13-19.17] | 22.85 [22.85-22.86] |
| PRs + ggml | 128,000 | 2,028.7 [2,025.5-2,032.0] | 68.3 [63.2-73.3] | 63.18 [63.08-63.28] | 66.93 [66.55-67.30] |
| PRs + ggml, spec 3 / 0.7, pipeline-windows 2 | 4,096 | 968.2 [945.9-990.6] | 77.2 [74.6-79.8] | 4.26 [4.16-4.36] | 7.56 [7.35-7.77] |
| PRs + ggml, spec 3 / 0.7, pipeline-windows 2 | 32,768 | 1,707.5 [1,706.4-1,708.5] | 82.3 [81.1-83.5] | 19.23 [19.22-19.24] | 22.32 [22.26-22.38] |
| PRs + ggml, spec 3 / 0.7, pipeline-windows 2 | 128,000 | 2,040.8 [2,034.9-2,046.7] | 75.7 [73.2-78.1] | 62.81 [62.62-62.99] | 66.17 [66.10-66.24] |

What the table shows:

- **0.1.41 against 0.1.40.1, stock split** (that report's rows, GFXOFF disabled there): 447 → 464 at 4K, 723 → 767 at
  32K, 821 → 869 at 128K (+4 / +6 / +6%); decode 62.1 / 67.0 / 59.0 → 62.6 / 65.0 / 60.8 (within the spread).
- **The two switches**: 791 / 1,443 / 1,632 → 832 / 1,619 / 1,823 (+5 / +12 / +12% over 0.1.40.1 with the same
  switches), so part of what the pull requests added on 0.1.40.1 is in the 0.1.41 release now. Decode 69 / 75 / 68.
- **The pull requests + ggml**: 968 / 1,714 / 2,029, the same as on 0.1.40.1 (968 / 1,685 / 2,003); +16 / +6 / +11%
  over the switches on 0.1.41. The remaining gap at 4K is #1167 (the GDN projection shapes), at 128K #1151 and the MMQ.
- **`STRATA_HIP_ADAPT_KERNEL_COPY=1`** costs nothing measurable (464 / 767 → 461 / 767; decode 62.6 / 65.0 → 61.5 /
  65.1) and the stock split read its 128K prompts without a stall at the GFXOFF default, which it did not in the
  0.1.40.1 report.
- **Decode**: the engine version and the pull requests leave it at 61-75 tok/s, as in the two earlier reports. `--spec 3 --spec-min-p 0.7 --pipeline-windows 2` on the PRs tree: 64.7 → 77.2 at 4K, 68.8 → 82.3 at 32K, 68.3 → 75.7 at 128K (+19 / +20 / +11%), prompt speed unchanged (±0.7%), draft acceptance 83% instead of 68% (`results.json`, `drafts_accepted / drafts_offered`). Measured on 2026-10-08 as well with the vision encoder loaded (three server starts, two runs each at 4K and 32K): 66 → 73-77 and 74 → 80-83. Each knob alone had been neutral or worse on this machine (`--pipeline-windows 2` with the default draft +1-6%; `--spec 3 --spec-min-p 0.7` without it flat to -3%; a 16-point spec × min-p grid at `--pipeline-windows 1` found 4 / 0.5 best): the shorter draft is accepted almost whole, so the speculative second window hits. See the limitations for what it does to reproducibility.
- **One card and helper against 0.1.40.1**: stock 458 / 473 / 459 → 473 / 502 / 486 (+3 / +6 / +6%), the two
  switches 811 / 973 / 913 → 850 / 1,102 / 1,027 (+5 / +13 / +12%), the pull requests 969 / 1,152 / 1,126 → 976 /
  1,174 / 1,147 (+1 / +2 / +2%) - the same pattern as the split: part of what the pull requests added on 0.1.40.1 is in
  the 0.1.41 release. One-card decode is 46-52 in every arm (a 16 GB card holds about 4,500 expert slots, hit rate
  82-83%); the helper mode decodes 65-75, with the pull requests 72.5 / 75.2 / 75.1 (the 0.1.40.1 report had 72 / 72 /
  66), reading prompts at one card's speed.
- Prompt readings at 32K and 128K are within 0.3% between the two runs in every arm; decode spreads up to 10% at 4K
  with draft acceptance.

### A note on the layer split and decode

This machine's production configuration also loads the vision encoder on the second card. With it, 0.1.41's split
search lands at layers 0-26 / 27-47 (the second card has 1.7 GiB less free VRAM) and the same binary decodes a 32K
prompt at 74 tok/s (three runs on 2026-10-08, 73.4-75.0); without it, as in every arm here, the split is 0-24 / 25-47,
the second card's expert cache is larger (4,829 slots instead of 3,968, hit rate 96.6% instead of 95.6%) and decode at
32K is 68.8. The 4K decode is the same either way. So the split point moves decode by about 8% at 32K on this
machine, independently of the cache hit rate; it was not swept here (`layer_split` can be given by hand).

### Reused prefix: an observation, not an arm

The arms above read every prompt in full. docs/COMMUNITY_BENCHMARKS.md asks for the warmed-cache and reused-prefix
conditions to be kept apart; they were not measured as arms here (the second run at each length is on a warmed expert
cache, the first 4K run is the only cold one). What a reused prefix does on this configuration is visible in the engine's
own accounting of one real agent session on 2026-10-09 (the production server, the same binary and decode configuration,
with the vision encoder loaded, `reasoning_effort` at the template's default, a coding agent as the client): eight
requests growing from 8.8K to 93.7K tokens of context, 0-99% of each prompt reused from the conversation cache,
the freshly read remainder (122-26,627 tokens) at 970-1,550 tok/s, decode 70.5-85.5 tok/s on replies of 287-1,960
tokens and **97.1 tok/s over a 14,821-token reply** at 93.7K context (expert cache hit rate 94%, 88% of drafts
accepted). Throughput numbers only; the session's content is not part of this report.

## Stalls, memory, power

- Zero stalls in the five arms (`stall report` / `timed out at layer` / `no progress for` in each `engine.log`: 0),
  including the 128K requests of the stock tree with `STRATA_HIP_ADAPT_KERNEL_COPY=1`. The GFXOFF knob read 1 (the
  default) at every start.
- VRAM peak 15.9-16.1 GiB on both cards (`mem_info_vram_used` at the end of each arm: 16,011-16,094 MiB on the first
  card, 15,887-15,930 on the second).
- RAM and power as in the 0.1.40.1 report (sampled once a second; the telemetry files are in the companion repository).

## Limitations

- One machine, one quantization, one context size, a small synthetic greedy workload with a 256-token output cap;
  no recall, vision, concurrency, sampling or thermal run. Warmed-cache and reused-prefix conditions were not run as
  separate arms (the observation above is from production use, not from the harness).
- The one-card and helper arms ran four hours after the split arms, after the box had run other work in between;
  the stock split's 4K / 32K rows and the one-card rows are the only two-way check across the two sessions.
- The "PRs" arm bundles the three open pull requests with two local commits (the rocBLAS solution cache) and a
  modified ggml; only the bundle was measured here. The pull requests do not change attention or selection values
  beyond FP32 summation order; the FP16 prompt route does (checked in
  [2026-10-04-rdna2-fp16-prompt](../2026-10-04-rdna2-fp16-prompt/README.md)).
- `--pipeline-windows 2` makes the output differ between server starts on this machine (the same request, greedy,
  produced different texts in 12 of 12 pairs across three starts; the baseline configuration reproduces its text
  across starts). Users who need reproducible greedy output should leave it off.
- The 0.1.40.1 numbers quoted for comparison are from this machine's 2026-10-06 run, not re-measured; that run had
  GFXOFF disabled through debugfs for the two-card arms, this one did not need it.

Developed with an AI coding assistant; every number above was measured on 2x RX 6900 XT (gfx1030, PCIe 4.0 x8 each) /
Ryzen 5 5600X, ROCm 10.0. The tuning history behind the "PRs" arm, step by step with its measurements, is at
[Vop3p/dual-6900xt-flash-next-tuning](https://github.com/Vop3p/dual-6900xt-flash-next-tuning).
