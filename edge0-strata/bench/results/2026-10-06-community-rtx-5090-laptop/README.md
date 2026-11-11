# Community benchmark: RTX 5090 Laptop GPU (24 GB), Ryzen 9 9955HX3D

Measured on 2026-10-06 by [wolffahrer](https://github.com/wolffahrer) on a Windows 11 gaming laptop
(Schenker XMG NEO 16, A25). This tests Strata 0.1.40.1 (engine 0.1.40) with Flash-Next **Q2_0**, one mobile GPU, and a
131,072-token context limit. It reuses the workload of the
[2026-09-30 desktop RTX 5090 report](../2026-09-30-community-rtx-5090/README.md) (same synthetic prompts, prompt lengths,
greedy decoding, reasoning off, 256-token output cap) so the two can be read side by side. It is **not** a controlled
comparison: quantization, engine version, OS, VRAM, power limit and PCIe link all differ (see below).

Median decode throughput was **136.9 tok/s at 4,096 prompt tokens, 137.7 tok/s at 32,768, and 134.6 tok/s at
128,000**. Median prompt throughput was 1,952.5, 2,877.8 and 2,764.1 tok/s. All nine speed requests succeeded with zero
reused tokens. These are synthetic code-explanation requests; they do not establish general answer quality.

## Hardware and software

- NVIDIA GeForce RTX 5090 **Laptop** GPU; 24,463 MiB reported VRAM. Power: current limit 175.00 W (default 95.00 W,
  max 175.00 W, laptop Dynamic Boost); sampled draw during the run median 162.6 W, peak sample 191.8 W.
  **PCIe link: Gen 4, x8** under load (Gen 2 x8 when idle; `nvidia-smi` reports x16 as the maximum width).
  The engine's startup transfer probe was not recorded. GPU clocks were not fixed.
- AMD Ryzen 9 9955HX3D, 16 cores / 32 logical CPUs. The run used `--pool-workers 10` (10 expert-pool workers plus the
  host thread on core 0), a value chosen by a local manual sweep.
- 64 GB installed RAM (2 x 32 GB DDR5-5600); Windows reported 61.68 GiB usable.
- Storage: model files and pack on a Samsung SSD 990 PRO 1TB (NVMe); OS on a WD Blue SN5100 500GB (NVMe).
- Windows 11 Home, build 26200. NVIDIA driver 610.88 (CUDA UMD 13.3).
- Strata **v0.1.40.1**, tag commit `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`, built from the release source archive
  (not a git checkout). Engine 0.1.40, compiled locally for CUDA architecture 120 with CUDA toolkit 13.3 (nvcc
  13.3.73), GPU vision helper included. See [BUILD.json](BUILD.json).
- Background: a WSL2 VM (memory cap 32 GB) with an idle local agent stack kept running; the regular local inference
  server was stopped for the run, so the GPU was dedicated to Strata and its vision helper. Not an isolated OS.

## Model and configuration

- Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, **Q2_0**:
  `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`, `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`,
  `mmproj-Qwen3.8-Flash-Next-BF16.gguf`. Repository revision and file hashes: **not measured** for this report.
  (The desktop report used IQ2_XS.)
- Pack and MTP draft experts prepared by the unmodified installer (`setup.py --build`); bundled expert profile.
- Context 131,072; INT8 KV; 32,768 KV cells per attention layer resident on the GPU (K/V in 1.55 GiB pinned RAM).
- Expert cache `auto`: **11,742 slots, 15.12 GiB VRAM** (the desktop had 17,463 slots / 23.44 GiB), profile-prefilled,
  no eviction. `--vram-reserve-mib 1000`; 497 MiB VRAM free with everything loaded. `--mmap-experts`.
- Prefill `auto`: 8,192-token chunks, borrowing 2,082 expert-cache slots (2.68 GiB) during prompt processing.
- MTP `--spec 4 --spec-min-p 0.7` (desktop: 0.5) and `--pcie-frac 0`, both from a local manual sweep; built-in suffix
  drafting enabled. No built-in calibration run, experimental speed projection off, low-RAM mode off.
- GPU vision enabled (1,024 image tokens); benchmark requests contained no images.
- Opt-ins that were on but do not affect these requests: conversation cache `--conversation-cache-mib 6144
  --conversation-cache-slots 2`, `reasoning_budget_tokens: 12000`, `reasoning_loop_recovery: "stop"`.
- Speed runs: reasoning off (`reasoning_effort: none`), temperature 0, at most 256 generated tokens.

The complete configuration (local paths replaced by placeholders) is [strata-q2_0.json](strata-q2_0.json); startup
choices and every request's timing and draft statistics are in [engine.log](engine.log).

```text
.venv\Scripts\python.exe serve\server.py --engine strata --config strata-q2_0.json --port 18096
```

## Method

[benchmark.py](benchmark.py) is the desktop report's script with three changes: it runs with Strata's own Windows
venv, a failed or cancelled request is recorded with its error and excluded from the statistics instead of stopping
the suite, and peak host RAM (total minus available, system-wide) and GPU memory are sampled every 0.5 s during each
request. Prompts, nonces, token targets and the token-count check against the server are unchanged.

```text
.venv\Scripts\python.exe benchmark.py --root <strata> --pack <strata-data>\packs\q2_0 --url http://127.0.0.1:18096 --out results
.venv\Scripts\python.exe tools\needle_bench.py --url http://127.0.0.1:18096 --lengths 32k,128k --depths 10,50,90 --out needles.json
.venv\Scripts\python.exe coding_check.py --url http://127.0.0.1:18096 --out coding-check.json
```

One short warm-up request was excluded. Three runs at each length ran serially in increasing-length order on the same
loaded engine. All nine speed requests processed their entire prompt (**zero reused tokens**). The expert cache was
filled at startup and kept between requests; no restart or page-cache reset separated them. Loading time is excluded
(the server was healthy 15.1 s after start). Throughput comes only from the engine's timing values: prompt tok/s =
freshly read tokens / `prompt_ms`, decode tok/s = `engine_generated / decode_ms`. TTFT is the client's streaming time
to the first non-empty text delta (answer text, reasoning off); total is until stream end. Both include HTTP over
loopback and frontend tokenization.

## Results

Median **[minimum–maximum]** of three runs. Every request generated 256 tokens and stopped at the output limit; no
failed or cancelled requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,952.5 [1,746.8–1,955.5] | 136.9 [125.0–137.4] | 2.124 [2.117–2.376] | 3.977 [3.977–4.430] |
| 32,768 | 0 | 2,877.8 [2,867.3–2,900.1] | 137.7 [134.2–140.0] | 11.458 [11.367–11.509] | 13.352 [13.184–13.357] |
| 128,000 | 0 | 2,764.1 [2,755.1–2,797.3] | 134.6 [122.1–141.9] | 46.498 [45.973–46.666] | 48.289 [47.863–48.751] |

Raw per-run records: [results.json](results.json) (engine timings, draft statistics, output text, request hashes,
per-request memory peaks). Aggregates: [summary.json](summary.json). Decode expert-cache hit rates were 90.2–98.5% for
the nine requests (the first 4K run, which also has the lowest decode speed, is retained in the range).

**Side by side with the desktop report** (not a controlled comparison): decode reached 76% (4K), 78% (32K) and 82%
(128K) of the desktop's medians (179.4 / 175.7 / 165.0 tok/s); prompt throughput reached 46%, 52% and 48%
(4,269.8 / 5,543.2 / 5,778.7 tok/s). Plausible contributors, none isolated here: 175 W vs 575 W power limit, PCIe
Gen 4 x8 vs Gen 5 x16, 24 vs 32 GB VRAM (15.1 vs 23.4 GiB expert cache), Q2_0 vs IQ2_XS, Windows vs Linux,
engine 0.1.40 vs 0.1.29, `--spec-min-p` 0.7 vs 0.5.

Memory, from [monitor.py](monitor.py) sampling once per second from before model load until the end of the checks
([telemetry.jsonl](telemetry.jsonl), 369 samples over 397.5 s, summary in [memory-summary.json](memory-summary.json)):

- GPU memory: **23,219 MiB peak** (22.67 GiB), system-wide on the single GPU.
- Host RAM in use (total minus available, system-wide, includes the WSL2 VM and other processes; not process RSS):
  23.63 GiB before load, **55.59 GiB peak**.
- Swap/page file in use: 0.64 GiB, unchanged during the run. Page-fault rates were not measured. No out-of-memory error.

## Correctness and limitations

- **Long-context recall:** the unchanged `tools/needle_bench.py` found **all six needles** (depths 10/50/90% at `32k`
  and `128k`). Actual prompt lengths 32,308–32,309 and 125,703–125,705 tokens. Two of the 128K cases and one 32K case
  reused 16,384 prompt tokens; recall latency is therefore not part of the speed table. See [needles.json](needles.json).
- **Coding check:** [coding_check.py](coding_check.py) asks for `roman_to_int(s)` with error handling and runs a fixed
  local test suite on the returned code: **10/10 tests passed** (7 conversions, 3 expected `ValueError`s), reasoning
  effort `low`, 1,160 generated tokens, 10.6 s. See [coding-check.json](coding-check.json).
- Not evaluated: long outputs, sampled decoding, thinking speed, vision, tool use, concurrency, a sustained thermal run.
  Laptop GPUs can change clocks with temperature and power state; GPU temperature peaked at 83 °C in this short run.
  The 128,000-token prompt does not fill the full 131,072-token window.

The author documents local AI on consumer hardware (in German) on the KI SOUVERÄN channels:
[YouTube](https://www.youtube.com/channel/UCE6Ch6g6Bo8v4ROpYDzOOxA) and [Telegram](https://t.me/lokale_ki).
