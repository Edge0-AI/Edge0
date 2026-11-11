# Community benchmark: 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), Ryzen 5 5600X, 128 GB RAM - Strata 0.1.40.1

Measured on 2026-10-06 on a Linux desktop. This tests the original Qwen3.8-Flash-Next (GSQ-RCO IQ3_S) with a
131,072-token context in the three ways two cards can be used - one card alone, a layer split across both, and one card
with the other as an expert helper (`--remote-expert-opt`) - on engine 0.1.40.1 as released, on 0.1.40.1 with the two
switches it ships for this card family (`STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1`), and on 0.1.40.1 with the three
open gfx103x pull requests #1150, #1151 and #1167 applied on top of those switches. The adaptive expert swaps were
**on** (the default) in every configuration; the 0.1.39 report from this machine had them off because of #884.

On stock 0.1.40.1, one card reads a prompt at about **460 tok/s** and decodes at 47-48 tok/s; with the two switches the
same card reads at **811 / 973 / 913 tok/s** (4K / 32K / 128K) and decodes at 49-51; with the pull requests it reads at
**969 / 1,152 / 1,126**. The layer split with the pull requests reads a 128,000-token prompt at **2,003 tok/s** (64 s
to the first token, against 156 s on the stock split and 279 s on one stock card) and decodes at 70-75 tok/s; the helper
mode decodes at 64-75 tok/s in every arm. All 54 recall checks (six needles per configuration) were found. The requests
are synthetic code-explanation requests with greedy decoding and a 256-token output cap; they do not establish answer
quality or performance on other workloads.

**One thing to know before reading the two-card numbers:** with the adaptive swaps on and the GPU's GFX power-gating
(GFXOFF) at the kernel's default, the stock and the switches-only layer split both stalled on their first 128,000-token
request (the engine stopped making progress while reading the prompt; the server restarted it). The two-card
configurations below were therefore measured with GFXOFF disabled through the amdgpu debugfs knob, which on this machine
removes the stall and does not change speed (the stalled runs' 4K and 32K rows are kept in the `*-gfxoffon` folders and
match the GFXOFF-off rows to the token). The one-card configurations ran at the default and did not stall. Details in
"Stalls" below; the issue is #884.

## Hardware and software

- **GPUs:** 2x AMD Radeon RX 6900 XT 16 GB (Navi 21, gfx1030, wave32, no matrix cores), reference cards, 245 W power
  cap, no clock changes, no display attached. Both on CPU PCIe 4.0 lanes of an X570 board, which gives each card x8 (the
  engine's own probe: 13.6-14.1 GB/s host to device per card). No P2P between the cards is used by any mode below.
- **CPU and RAM:** AMD Ryzen 5 5600X (6 cores / 12 threads, AVX2, no AVX-512; the engine ran 5 expert-pool workers plus
  its host thread). 128 GB DDR4-3200 (4 DIMMs). 8 GiB swap file.
- **Storage:** the model files and the pack on one NVMe SSD (Samsung 970 EVO 500 GB), also the system disk. The PLE
  table is read from the GGUF shard on it (`--ple-io direct`, the default).
- **OS and runtime:** Ubuntu 26.04, kernel 7.0.0-38-generic with its own amdgpu driver (no DKMS), ROCm 10.0.0 (HIP
  7.15, rocBLAS 5.6.0.8d1ae90e). hipBLASLt ships no gfx1030 kernels, so the plain hipBLAS/rocBLAS path runs.
- **Source, stock:** [Niko1221/Strata](https://github.com/Niko1221/Strata) `main` at `82f46a8` (engine 0.1.40.1).
- **Source, "PRs":** `82f46a8` with #1150 (PR #540's attention kernel on gfx103x, bit-exact), #1151 (the QSA block
  scores on rocBLAS SGEMM, opt-in `STRATA_SELECT_SGEMM=1`) and #1167 (a rocBLAS solution table for the FP16 prompt
  GEMMs, `STRATA_ROCBLAS_TUNING`) merged locally, plus #1148 (a test skip) and #1149 (tests and an off-by-default
  range counter for the FP16 route); the tree is `341922e` in each `BUILD.txt`. The engine source of #1148 and #1149
  does not change any number here.
- **Build:** both trees built by hand with cmake and ROCm's clang: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_PREFILL_MMQ=ON` (the options setup uses for AMD), llama.cpp at the pinned
  commit. Each configuration's `BUILD.txt` has the tree, the commit, the binary's SHA-256, the environment and the
  GFXOFF knob's reading at start.
- **Other load:** nothing else used the GPUs during the runs (telemetry, below). The machine's other services (an idle
  llama-swap router, a home-automation stack, the LACT fan-curve daemon) kept running on the CPU.

## Model and configuration

**Model:** [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
IQ3_S - the same two files setup downloads for its IQ3_S choice, packed with the repository's `tools/iq_pack.py` (no
`--compat-bf16`); the SHA-256 hashes match the repository's `.sha256` files (they are listed in the 2026-10-05 report
from this machine). The MTP draft layer was prepared with the repository's tools. The expert profile is the repository's
own `data/expert-profile.bin`. No vision encoder, no calibration, no speed projection.

**Configurations** (each one's `config.json` is in its folder; `/m` is the model directory, `/work/stock` and
`/work/prs` the two trees). Common to all: context 131,072, `--kv int8 --kv-resident 32768` (32,768 K/V cells per
attention layer in VRAM, the rest streamed from pinned RAM), `--expert-cache auto`, `--prefill auto` (8,192-token
chunks), MTP with `--spec 4 --spec-min-p 0.5`, suffix drafting on, and the adaptive expert swaps at their default
(`--adapt-every 4`, not passed).

| name | cards | engine | environment |
| --- | --- | --- | --- |
| `single-stock` | one (PCI 0f) | stock | none |
| `single-sw` | one | stock | `STRATA_HIP_PROMPT_F16=1 STRATA_SH_STREAM=1` |
| `single-prs` | one | PRs | the two above + `STRATA_ROCBLAS_TUNING=/work/prs/tools/hip STRATA_SELECT_SGEMM=1 STRATA_SELECT_SGEMM_VERBOSE=1 STRATA_IO_THREADS=128` |
| `split-stock` | both, `layer_split: auto` | stock | none |
| `split-sw` | both | stock | the two switches |
| `split-prs` | both | PRs | as `single-prs` |
| `helper-stock` | one + the other as helper (`--expert-cache-device1 auto --remote-expert-opt`), the probed PCIe share 0.39 | stock | none |
| `helper-sw` | the same | stock | the two switches |
| `helper-prs` | the same | PRs | as `single-prs` |
| `split-stock-gfxoffon`, `split-sw-gfxoffon` | both | stock | as the two split arms above, GFXOFF at the kernel default; stalled at 128K (below) |

The engine's own log confirms each switch: `STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out`
(once per card), `prefill gemm: rocBLAS tuning enabled (112 rows, 76 with a solution, gfx1030, rocBLAS 5.6.0.8d1ae90e)`
and the `qsa select: SGEMM reach<=... ` measurement lines (`engine.log` in each folder).

**GFXOFF.** `/sys/kernel/debug/dri/{0,1}/amdgpu_gfxoff` (4 bytes, little-endian) was 1 (the default) for the three
one-card arms and for the two `*-gfxoffon` arms, and 0 for the six two-card arms, written as root before `split-stock`
and written back to 1 after `helper-prs`; the knob's reading at each start is in `BUILD.txt`. The knob is a reference
count in amdgpu (`amdgpu_gfx_off_ctrl`): a 0 is undone only by a later 1.

## Method

[benchmark.py](benchmark.py) is the script of the
[RTX 5090 report](../2026-09-30-community-rtx-5090/README.md), unchanged. It builds deterministic synthetic Python
filler, puts a different nonce near the start of each request, and counts the rendered chat prompt with Strata's
tokenizer. [monitor.py](monitor.py) samples RAM, swap and both cards' VRAM, GTT, busy percentage, power and temperature
from sysfs once per second.

Every configuration started its own server (`serve/server.py --engine strata --config <config.json> --port 8089` from
its tree), and ran, in this order:

```bash
python3 benchmark.py --root /work/<tree> --pack /m/packs/iq3_s --url http://127.0.0.1:8089 --out <folder> --targets 4096,32768,128000 --runs 3
python3 /work/<tree>/tools/needle_bench.py --url http://127.0.0.1:8089 --lengths 32k,128k --depths 10,50,90 --out <folder>/needles.json
```

- One short warm-up request is excluded.
- Three runs at each length ran serially, in increasing-length order, on the same loaded engine.
- All 81 speed requests of the nine complete configurations read their whole prompt: **zero reused tokens**; every one
  generated 256 tokens and stopped at the output cap (`results.json`).
- The expert cache was filled from the profile at start and kept between requests. Loading time is excluded.
- TTFT is streaming, from just before the HTTP request to the first nonempty text delta, over loopback.
- Prompt throughput is freshly read tokens / `prompt_ms`; decode throughput is the engine's `decode_tok_s`, both from
  the engine's own accounting.
- `reasoning_effort: none`, temperature 0.
- Each folder has the engine's whole log for its session (`engine.log`), the benchmark's output, `results.json`
  (request hashes and every output text), `summary.json`, `needles.json` and `needles.out`, `telemetry.jsonl`, the
  server's status at the end and `BUILD.txt`.

The eleven servers ran one after the other between 14:19 and 17:51 local time: the three one-card arms, then the two
split arms that stalled (15:30-15:56), then the six two-card arms with GFXOFF disabled.

## Results

Each cell is the median **[minimum-maximum]** of three runs. No speed request of the nine complete configurations
failed, was cancelled or stalled.

### One card (PCI 0f), GFXOFF default

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 4,096 | 458.0 [451.4-458.8] | 47.5 [42.2-49.3] | 8.98 [8.97-9.11] | 14.33 [14.15-15.15] |
| stock | 32,768 | 473.4 [472.9-473.7] | 46.9 [46.5-50.7] | 69.30 [69.26-69.36] | 74.77 [74.28-74.79] |
| stock | 128,000 | 458.5 [458.3-458.9] | 48.3 [45.0-48.3] | 279.37 [279.10-279.51] | 284.77 [284.37-285.03] |
| switches | 4,096 | **810.9** [780.2-811.5] | 50.5 [45.1-54.9] | 5.09 [5.09-5.29] | 10.13 [9.72-10.94] |
| switches | 32,768 | **973.3** [971.9-974.2] | 48.5 [48.3-51.1] | 33.74 [33.71-33.79] | 38.99 [38.72-39.04] |
| switches | 128,000 | **912.8** [912.1-913.0] | 51.3 [49.8-52.7] | 140.42 [140.40-140.53] | 145.38 [145.23-145.63] |
| PRs | 4,096 | **969.4** [954.8-982.8] | 51.8 [44.6-54.2] | 4.26 [4.21-4.33] | 9.12 [8.96-10.04] |
| PRs | 32,768 | **1,152.4** [1,150.7-1,152.7] | 50.9 [50.1-54.3] | 28.51 [28.50-28.55] | 33.50 [33.20-33.63] |
| PRs | 128,000 | **1,125.6** [1,123.2-1,126.2] | 50.0 [49.2-50.1] | 113.92 [113.86-114.16] | 119.03 [119.01-119.24] |

### Layer split across both cards, GFXOFF disabled

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 4,096 | 446.5 [439.2-447.3] | 62.1 [61.9-70.0] | 9.21 [9.19-9.36] | 13.29 [12.84-13.48] |
| stock | 32,768 | 723.4 [721.6-724.5] | 67.0 [61.1-69.4] | 45.37 [45.30-45.48] | 49.28 [48.96-49.53] |
| stock | 128,000 | 821.2 [820.7-821.9] | 59.0 [58.2-59.1] | 156.06 [155.93-156.16] | 160.43 [160.24-160.46] |
| switches | 4,096 | **790.9** [752.2-793.7] | **70.4** [67.6-75.3] | 5.21 [5.20-5.48] | 8.81 [8.59-9.24] |
| switches | 32,768 | **1,443.3** [1,442.6-1,443.3] | **73.3** [69.1-75.4] | 22.77 [22.77-22.78] | 26.24 [26.15-26.46] |
| switches | 128,000 | **1,631.5** [1,629.7-1,634.1] | **64.7** [62.2-68.6] | 78.65 [78.52-78.73] | 82.61 [82.36-82.66] |
| PRs | 4,096 | **968.0** [945.7-971.4] | **72.2** [62.2-78.7] | 4.26 [4.25-4.36] | 7.79 [7.48-8.46] |
| PRs | 32,768 | **1,684.7** [1,682.5-1,688.8] | **74.7** [69.6-77.5] | 19.52 [19.47-19.54] | 22.92 [22.82-23.13] |
| PRs | 128,000 | **2,003.1** [2,001.0-2,005.9] | **70.3** [68.9-72.1] | 64.09 [64.00-64.16] | 67.68 [67.62-67.78] |

The two stalled runs at the GFXOFF default, 4K and 32K rows only (`split-stock-gfxoffon`, `split-sw-gfxoffon`):
stock 447.2 / 723.3 tok/s prompt and 62.1 / 67.0 decode; switches 782.2 / 1,446.6 and 70.5 / 73.5 - the same as with
GFXOFF disabled.

### One card with the other as expert helper (probed PCIe share 0.39), GFXOFF disabled

| | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 4,096 | 459.2 [452.6-459.9] | 71.3 [63.2-73.1] | 8.95 [8.94-9.08] | 12.52 [12.42-13.11] |
| stock | 32,768 | 473.6 [473.4-473.7] | 66.4 [65.3-68.7] | 69.26 [69.25-69.29] | 73.12 [72.96-73.15] |
| stock | 128,000 | 458.7 [458.5-458.7] | 63.9 [63.3-70.3] | 279.25 [279.22-279.38] | 283.27 [282.83-283.36] |
| switches | 4,096 | **816.5** [780.0-819.6] | 74.9 [62.3-80.8] | 5.05 [5.03-5.28] | 8.44 [8.17-9.37] |
| switches | 32,768 | **975.2** [973.4-975.8] | 71.5 [69.8-71.6] | 33.67 [33.65-33.73] | 37.23 [37.21-37.38] |
| switches | 128,000 | **912.8** [912.7-912.8] | 68.7 [68.3-69.2] | 140.42 [140.41-140.44] | 144.13 [144.09-144.16] |
| PRs | 4,096 | **979.7** [962.5-980.5] | 72.1 [66.1-76.5] | 4.21 [4.21-4.29] | 7.74 [7.54-8.14] |
| PRs | 32,768 | **1,151.0** [1,150.3-1,152.8] | 72.1 [71.3-77.7] | 28.54 [28.49-28.56] | 32.02 [31.81-32.13] |
| PRs | 128,000 | **1,126.9** [1,126.8-1,127.1] | 65.5 [64.2-66.8] | 113.78 [113.76-113.79] | 117.67 [117.60-117.72] |

What the tables show:

- **The two shipped switches** read prompts 1.8x faster at 4K and 2.0x at 32K and 128K on one card (the FP16 prompt
  GEMMs, #835 in 0.1.40), 1.8-2.0x on the split, and decode 5-13% faster (the shared-expert stream fork, off by default
  in 0.1.40, `STRATA_SH_STREAM=1`): one card 47.5 -> 50.5, split 62.1 -> 70.4 and 67.0 -> 73.3, helper 71.3 -> 74.9.
  The 128K first-token time on one card goes from 279 s to 140 s.
- **The pull requests** add 20-22% at 4K (the rocBLAS solution table, #1167: the GDN projection below 1,152 tokens) and
  17-23% at 32K and 128K (the attention kernel #1150 and, at 128K, the SGEMM block scores #1151): one card
  913 -> 1,126 tok/s at 128K, the split 1,631 -> 2,003. Decode is within 3% of the switches-only arm (the kernels are
  prompt-path kernels).
- **Against the 2026-10-05 report** (0.1.39, adaptive swaps off), decode is 8-18% higher: one stock card 40-43 ->
  47-48 tok/s, the helper mode with the PRs 60-65 -> 65-72. That is the swaps and the 0.1.40 changes together; the two
  were not separated here.
- **The layer split** reads long prompts faster than one card (each card runs its own layers; 2,003 against 1,126 at
  128K with the PRs) but 4K prompts no faster (968 against 969), and decodes 62-75 against 47-52. **The helper mode**
  reads prompts at one card's speed (only the primary reads prompts) and decodes 64-75, between the two.
- Run-to-run spread: the three prompt readings at 32K and 128K are within 0.4% of each other in every configuration;
  decode varies more (up to 20%, the 4,096-token rows most) with draft acceptance, although the text is greedy.
- The decode expert-cache hit rate per request (each `engine.log`) was 28-83% on one card, 59-98% on the split and
  48-99% in the helper modes, lowest on the first requests.

**Memory and power**, from each `telemetry.jsonl` (one-second samples from the server's start through the recall
checks):

- VRAM peak: 15.5-15.7 GiB (15,887-16,131 MiB of `mem_info_vram_used`, the whole card) on every card that held weights.
- RAM: `MemAvailable` never below **65.6 GiB** of 121.5; swap: no growth (1.3 GiB from earlier use, constant).
- Power: a card running layers peaked at 249-254 W (`power1_average`; the cap is 245 W, which the average can
  overshoot briefly); the helper card at 75-79 W; an idle card at 13 W.

These are sampled values; brief peaks between samples can be missed.

## Stalls

With GFXOFF at the kernel's default, `split-stock` and `split-sw` each answered their 4K and 32K requests and then
stopped making progress on the first 128,000-token request while reading the prompt - `engine.log`:
`strata serve: no progress for 60 s during a request (reading the prompt (batched): finishing the chunk from token
24576)` (stock; 40960 for the switches arm), with the stall report showing the expert pool idle and no layer served -
and the server restarted the engine (#29); each arm stalled once more during its recall checks (chunk 65536 in the
switches arm), so one needle of six went unanswered there and `needles.out` counts 5 of 5. The same two configurations
with GFXOFF disabled, and the four other two-card arms, ran 54 speed requests and 36 recall checks without a stall;
the three one-card arms ran at the default without a stall. The 0.1.39 report from this machine avoided the stall by
turning the adaptive swaps off (`--adapt-every 100000`). The stall and the GFXOFF finding are in #884.

## Recall and limitations

The repository's unchanged `tools/needle_bench.py` found **all six needles** in each of the nine complete
configurations, at depths 10%, 50% and 90% at both lengths (`needles.json` in each folder): the `32k` prompts were
32,156-32,309 tokens and the `128k` prompts 125,510-125,705 tokens, all read in full (no reused prefix).

**Limits of this report:**

- One machine, one quantization (IQ3_S), one context size and a small synthetic workload. Long output, sampled
  decoding, thinking, coding-task correctness, vision, tool use, concurrency and a sustained thermal run were not part
  of it.
- The two-card numbers are with GFXOFF disabled (above); the one-card numbers at the default. On this machine the
  setting did not change any measured speed, but it is a non-default kernel state.
- The "PRs" numbers are for pull requests that were open when this was measured. #1150 and #1151 do not change the
  attention or selection values beyond FP32 summation order; the FP16 prompt route (`STRATA_HIP_PROMPT_F16=1`, in
  0.1.40 as an opt-in) does, and its distribution check is in
  [bench/results/2026-10-04-rdna2-fp16-prompt](../2026-10-04-rdna2-fp16-prompt/README.md).
- The 0.1.39 numbers quoted for comparison are from this machine's 2026-10-05 run of the same script (stock `6f32ec0`,
  adaptive swaps off), not re-measured today.
