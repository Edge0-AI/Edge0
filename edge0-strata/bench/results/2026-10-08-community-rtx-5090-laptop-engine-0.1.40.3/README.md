# Community benchmark: RTX 5090 Laptop GPU (24 GB), engine 0.1.40.3

Measured in the night of 2026-10-07/08 by [wolffahrer](https://github.com/wolffahrer) on the same Windows 11 gaming
laptop (Schenker XMG NEO 16, A25) as [2026-10-06-community-rtx-5090-laptop](../2026-10-06-community-rtx-5090-laptop/README.md)
(engine 0.1.40.1, PR #1263). This is a follow-up: same machine, same model files, same configuration, same scripts,
same order of requests. Only the Strata version changed. 0.1.40.2 was measured the same night with the identical
procedure; its numbers are in a short section below.

Median decode throughput with **0.1.40.3** was **138.1 tok/s at 4,096 prompt tokens, 144.4 tok/s at 32,768, and
131.0 tok/s at 128,000**. Median prompt throughput was 1,949.7, 2,893.3 and 2,780.1 tok/s. All nine speed requests
succeeded with zero reused tokens. Against 0.1.40.1 this is within run-to-run variation (see the comparison table).

## Hardware and software

Unchanged from the 0.1.40.1 entry unless noted:

- NVIDIA GeForce RTX 5090 **Laptop** GPU, 24,463 MiB reported VRAM, 175 W limit (laptop Dynamic Boost). Sampled draw
  during this run median 163.9 W, peak sample 179.4 W; GPU temperature peaked at 84 °C. PCIe Gen 4 x8 under load.
- AMD Ryzen 9 9955HX3D, `--pool-workers 10`. 64 GB RAM (61.68 GiB usable). Samsung SSD 990 PRO 1TB for model and pack.
- Windows 11 Home, build 26200. NVIDIA driver 610.88. CUDA toolkit 13.3.
- Strata **v0.1.40.3**, tag commit `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`, built from the release source archive
  (sha256 `6fdca77a04d8535c598c6602990789892d999000f8b7e956a27ed0dc602b5cba`) with the unmodified installer for CUDA
  architecture 120, GPU vision helper included. See [BUILD.json](BUILD.json).
- Background: a WSL2 VM with an idle local agent stack; the regular inference server was stopped, so the GPU was
  dedicated to Strata and its vision helper. Not an isolated OS.

## Model and configuration

Identical to the 0.1.40.1 entry: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` **Q2_0**, context 131,072, INT8 KV with
32,768 resident cells, expert cache `auto` (again **11,742 slots, 15.12 GiB VRAM**, 495 MiB VRAM free with everything
loaded), prefill `auto` (borrows 2,082 slots), MTP `--spec 4 --spec-min-p 0.7`, `--pcie-frac 0`, `--mmap-experts`, GPU
vision, conversation cache 6 GiB / 2 slots, `reasoning_budget_tokens: 12000`, `reasoning_loop_recovery: "stop"`.
Complete configuration with local paths replaced by placeholders: [strata-q2_0.json](strata-q2_0.json). Startup
choices and every request's timing are in [engine.log](engine.log).

## Method

Same scripts as the 0.1.40.1 entry: [benchmark.py](benchmark.py) (4K/32K/128K synthetic code-explanation prompts,
three runs each, greedy, reasoning off, 256-token cap), the unchanged `tools/needle_bench.py` (32k/128k at depths
10/50/90) and [coding_check.py](coding_check.py) (reasoning `low`). One warm-up request excluded. The server was healthy
12.1 s after start. Memory and GPU state were sampled once per second by [monitor.py](monitor.py); the summary is in
[memory-summary.json](memory-summary.json) (298 samples over 393.0 s; the raw `telemetry.jsonl` is left out to keep
the folder small).

```text
.venv\Scripts\python.exe serve\server.py --engine strata --config strata-q2_0.json --port 18096
.venv\Scripts\python.exe benchmark.py --root <strata> --pack <strata-data>\packs\q2_0 --url http://127.0.0.1:18096 --out results
.venv\Scripts\python.exe tools\needle_bench.py --url http://127.0.0.1:18096 --lengths 32k,128k --depths 10,50,90 --out needles.json
.venv\Scripts\python.exe coding_check.py --url http://127.0.0.1:18096 --out coding-check.json
```

## Results (0.1.40.3)

Median **[minimum–maximum]** of three runs. Every request generated 256 tokens; no failed or cancelled requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,949.7 [1,805.6–1,958.6] | 138.1 [125.5–148.5] | 2.133 [2.124–2.302] | 4.083 [3.848–4.329] |
| 32,768 | 0 | 2,893.3 [2,888.9–2,906.8] | 144.4 [140.1–145.3] | 11.384 [11.330–11.427] | 13.145 [13.145–13.177] |
| 128,000 | 0 | 2,780.1 [2,752.8–2,801.1] | 131.0 [123.2–140.4] | 46.246 [45.884–46.702] | 48.513 [47.827–48.540] |

Raw per-run records: [results.json](results.json); aggregates: [summary.json](summary.json). Decode expert-cache hit
rates were 90.2–98.7% for the nine speed requests. Peak GPU memory 23,221 MiB; host RAM in use (system-wide, includes
the WSL2 VM) 12.74 GiB before load, 54.88 GiB peak; page file unchanged at 0.50 GiB.

**Across versions on this laptop** (median decode / prompt tok/s; same procedure, not interleaved, one run of three
requests per length each):

| Prompt tokens | 0.1.40.1 (2026-10-06) | 0.1.40.2 (2026-10-07 23:16) | 0.1.40.3 (2026-10-08 00:50) |
| ---: | ---: | ---: | ---: |
| 4,096 | 136.9 / 1,952.5 | 139.3 / 1,960.0 | 138.1 / 1,949.7 |
| 32,768 | 137.7 / 2,877.8 | 141.6 / 2,890.0 | 144.4 / 2,893.3 |
| 128,000 | 134.6 / 2,764.1 | 137.2 / 2,764.1 | 131.0 / 2,780.1 |

Decode differences of a few percent are inside the min–max ranges of single runs; prompt throughput is flat.
We read this as "no measurable speed change on this hardware" for 0.1.40.2 and 0.1.40.3, and no regression.

## Correctness and limitations

- **Long-context recall:** all six needles found (prompt lengths 32,895–32,896 and 125,868–125,870 tokens). One 32K and
  two 128K cases reused 16,384 prompt tokens. See [needles.json](needles.json).
- **Coding check (0.1.40.3):** **10/10 tests passed**, 823 generated tokens, 7.9 s. See [coding-check.json](coding-check.json).
- **Coding check (0.1.40.2), a note on greedy decoding:** the 0.1.40.2 run scored **0/10**. The answer text was
  empty after 2,567 generated tokens, all of them reasoning (the raw response of that run was not kept). We re-ran the
  check three times per version on a freshly started server and kept the raw responses: 0.1.40.2 scored 0, 10, 10 and
  0.1.40.1 scored 0, 10, 6. Both zeros were the first request after the start, with **identical** looping reasoning in
  both versions ("If 'IV' with invalid char? 'IV' pair. ..."), ended by the configured
  `reasoning_loop_recovery: "stop"` (`finish_reason: length`, 2,048 tokens). The result varies between runs in both
  versions, so we do not read the 0/10 as a 0.1.40.2 regression but as a property of greedy decoding with reasoning
  `low` on this Q2_0 model and prompt.
- Not evaluated: long outputs, sampled decoding, thinking speed, vision, tool use, concurrency, sustained thermal load.
  Laptop GPU clocks can vary with temperature and power state.

Besides these synthetic runs we checked both versions with our own agent task suite (multi-step tool use under a local
agent, German prompts, 44 tasks x 3 runs per version). Pass rates were within two runs of 0.1.40.1 on every set
(total 121, 122 and 123 of 132 runs for 0.1.40.1, 0.1.40.2 and 0.1.40.3); those tasks are private and not part of this
report.

The author documents local AI on consumer hardware (in German) on the KI SOUVERÄN channels:
[YouTube](https://www.youtube.com/channel/UCE6Ch6g6Bo8v4ROpYDzOOxA) and [Telegram](https://t.me/lokale_ki).
