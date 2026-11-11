# Community benchmark: RTX 4090 Laptop GPU, Core i9-13900HX

Measured on 2026-10-04 by [30crows](https://github.com/30crows) on a Lenovo Legion
notebook (model 82WQ). This tests Strata 0.1.38 with the original Flash-Next IQ3_S
and one GPU, with a 131,072-token context limit and, in a second session, a
262,144-token limit.

With the 131,072-token limit, median decode throughput was **51.2 tok/s at 4,096
prompt tokens, 49.9 tok/s at 32,768, and 48.1 tok/s at 128,000**; median prompt
throughput was 1,345 / 2,506 / 2,386 tok/s. With the 262,144-token limit, a
**250,000-token prompt was read at 2,116 tok/s (118.5 s to the first token) and
decoded at 40.3 tok/s**; at the shorter lengths decode was 13-17% slower than with
the 131,072-token limit ([below](#262144-token-context)). These are synthetic
code-explanation requests with greedy decoding and a 256-token output cap. They do
not establish general answer quality or performance on other workloads.

## Hardware and software

- NVIDIA GeForce RTX 4090 Laptop GPU (AD103); 16,376 MiB reported VRAM. nvidia-smi
  reported a 175 W current and maximum power limit (default 80 W), with
  `nvidia-powerd` (Dynamic Boost) active. PCIe capability: Gen 4, x16. The engine's
  startup transfer probe reported 19.3 GB/s host-to-device. GPU clocks were not fixed.
- Intel Core i9-13900HX; 32 logical CPUs. No AVX-512: the engine used AVX2 with
  23 expert-pool workers plus its host thread. CPU limits from the Lenovo firmware
  attributes: PL1 125 W, PL2 190 W.
- 64 GB installed RAM; Linux reported 61.28 GiB usable. Storage: WD_BLACK SN850X
  2 TB NVMe (model files and data directory). 512 GiB swap file.
- Platform profile `performance` (the `lenovo-wmi-gamezone` driver also offers
  `max-power` and `custom`; they were not used).
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, NVIDIA driver 610.57.04 (reports
  CUDA 13.3).
- Source commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`; engine 0.1.38.
  **Source build:** setup found no ready-made engine for v0.1.38 (the release
  download returned HTTP 404) and compiled one locally for CUDA architecture 89 with
  CUDA Toolkit 13.4 (NVCC V13.4.92) and GCC 15.2.0. Python 3.14.4. Hashes of the
  engine binary and the expert profile are in [artifact-hashes.json](artifact-hashes.json).
- During the run, Firefox and other large applications were closed; the desktop
  session stayed running. The machine was not otherwise isolated.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09` (setup's pinned revision):

- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54,817,524,224 bytes)
- `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (28,800,138,432 bytes)

The files were downloaded and checked by the unmodified installer; they were not
hashed again for this report. The MTP draft layer was fetched by the installer from
`Qwen/Qwen3.8-Flash-Next`. No vision encoder.

Setup command:

```bash
./setup.sh --yes --family qwen --model IQ3_S --context 131072 --kv int8 --vision no --no-start
```

- Context 131,072; INT8 KV. The launch arguments contain no `--kv-resident`, so KV
  streaming was off (setup's choice for this RAM size).
- Expert cache `auto`: 3,865 slots, 7.38 GiB VRAM, profile-prefilled, no eviction.
  The bundled expert profile was used without calibration.
- Prefill `auto` selected chunks of 8,192 tokens, borrowing 2,333 expert-cache
  slots (4.43 GiB) during prompt processing.
- 46.84 GiB of experts loaded into RAM at 4.96 GiB/s; 4 KB pages (no huge pages).
- MTP `--spec 4 --spec-min-p 0.5`.
- Low-RAM mode off; vision off; experimental speed projection off.
- Reasoning disabled, temperature 0, maximum 256 generated tokens per speed run.

The complete configuration is [strata-iq3_s.json](strata-iq3_s.json). The engine log
of the measured session (from engine start to the last recall check) is
[engine.log](engine.log). The server was started with `BROWSER=true ./run-iq3_s.sh`
so that it opened no browser.

## Method and reproduction

The benchmark and monitor scripts are the ones from the
[RTX 5090 report](../2026-09-30-community-rtx-5090/README.md), unchanged, run from
the repository root against the installed server:

```bash
B=bench/results/2026-09-30-community-rtx-5090
.venv/bin/python $B/monitor.py ~/strata-bench/telemetry.jsonl &
.venv/bin/python $B/benchmark.py --root . --pack ../Strata-data/packs/iq3_s \
  --url http://127.0.0.1:8080 --out ~/strata-bench
kill %1
.venv/bin/python tools/needle_bench.py --url http://127.0.0.1:8080 \
  --lengths 32k,128k --depths 10,50,90 --out ~/strata-bench/needles.json
```

Before the benchmark, the same loaded engine answered one 1,000-token request
(a power check, not part of the results). One short warm-up request was then
excluded. Three runs at each length were executed serially in increasing-length
order. All nine speed requests processed their entire prompt: **zero reused
tokens**. The GPU expert cache was filled at startup and kept between requests;
loading time is excluded.

TTFT and total latency are measured by the client over loopback. Engine prompt
throughput is freshly read tokens divided by `prompt_ms`; decode throughput is
`engine_generated / decode_ms`.

## Results

Each cell is the median **[minimum–maximum]** of three runs. Every request generated
256 tokens and stopped at the output limit, with no failed or cancelled requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,345.0 [1,280.0–1,367.7] | 51.2 [42.7–53.9] | 3.084 [3.032–3.235] | 8.004 [7.802–9.205] |
| 32,768 | 0 | 2,505.5 [2,496.1–2,511.8] | 49.9 [49.0–53.1] | 13.150 [13.135–13.201] | 18.305 [17.919–18.345] |
| 128,000 | 0 | 2,385.8 [2,376.5–2,412.7] | 48.1 [47.6–48.8] | 53.818 [53.217–54.043] | 59.164 [58.432–59.333] |

Raw per-run records: [results.json](results.json); aggregates:
[summary.json](summary.json). Decode expert-cache hit rates were 68.2–80.3%, and
the model accepted 136–162 of 203–234 MTP drafts per request. The lowest decode
run (42.7 tok/s) is the first 4K run, which also had the lowest hit rate (68.2%).

[telemetry.jsonl](telemetry.jsonl) has 267 samples over 273.5 seconds, covering the
speed suite only (not the recall checks). Observed peaks
([memory-summary.json](memory-summary.json)):

- GPU memory: **15,698 MiB** of 16,376 MiB, system-wide.
- Host RAM: **54.74 GiB** used (`MemTotal - MemAvailable`, system-wide, not process
  RSS); MemAvailable never fell below 6.55 GiB.
- Swap: 698.5 MiB allocated at the start and at the peak (no increase during the
  run). Swap I/O was not measured.
- GPU power: median **170 W** over the 250 samples with utilization above 50%, peak
  176 W; peak temperature 86 °C. Prompt processing ran at the power limit.

A separate power check on the same session (one 1,000-token answer, 25-token
prompt, not part of the table) decoded at 57.0 tok/s with a median GPU draw of
116 W (peak 141 W) over 39 busy samples: decode did not reach the power limit.

## Recall

The unchanged `tools/needle_bench.py` found **all six needles** at depths 10%,
50% and 90%. Actual prompt lengths were 31,969–31,970 tokens for `32k` and
122,753–122,755 for `128k`. See [needles.json](needles.json). The third case of
each length reused 16,384 prompt tokens (engine `/metrics`); the others reused zero.

## 262,144-token context

Measured later the same day (after a reboot) on the same machine, commit, engine
build, model files and power mode. Only the context changed:

```bash
./setup.sh --setup --yes --family qwen --model IQ3_S --context 262144 --kv int8 --vision no --no-start
```

The resulting configuration differs from the 131,072-token one only in
`--max-context 262144` ([context-262144/strata-iq3_s.json](context-262144/strata-iq3_s.json)).
KV streaming stayed off, so the larger INT8 KV cache sits in VRAM and the expert
cache shrank to **2,856 slots (5.47 GiB)** from 3,865 (7.38 GiB). Prefill still used
8,192-token chunks and borrowed 2,319 slots (4.43 GiB). The PCIe probe reported
20.6 GB/s. Engine log of this session: [context-262144/engine.log](context-262144/engine.log).

The same benchmark script ran with `--targets 4096,32768,128000,250000`, and the
monitor ran alongside it. No other request preceded the warm-up. All twelve speed
requests read their entire prompt (**zero reused tokens**), generated 256 tokens
and stopped at the output limit.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 1,343.0 [1,266.5–1,397.2] | 43.4 [37.4–44.1] | 3.097 [2.973–3.278] | 9.056 [8.961–9.783] |
| 32,768 | 0 | 2,457.5 [2,434.0–2,469.2] | 43.3 [43.0–43.3] | 13.412 [13.349–13.539] | 19.295 [19.228–19.457] |
| 128,000 | 0 | 2,291.6 [2,290.5–2,327.3] | 39.7 [39.5–42.6] | 56.027 [55.166–56.054] | 62.031 [61.612–62.448] |
| 250,000 | 0 | 2,116.4 [2,115.7–2,132.3] | 40.3 [40.2–41.0] | 118.457 [117.552–118.483] | 124.661 [123.891–124.798] |

Raw records: [context-262144/results.json](context-262144/results.json); aggregates:
[context-262144/summary.json](context-262144/summary.json). Decode expert-cache hit
rates were 67.1–74.6% (68.2–80.3% with the 131,072-token limit), and the model
accepted 134–155 of 203–246 MTP drafts per request. Against the 131,072-token
configuration, median decode was 15% lower at 4,096 tokens, 13% lower at 32,768
and 17% lower at 128,000; prompt throughput was within 4%.

The telemetry ([context-262144/telemetry.jsonl](context-262144/telemetry.jsonl),
662 samples over 679.5 seconds, speed suite only) shows peaks of **15,718 MiB** GPU
memory and **55.11 GiB** host RAM used; MemAvailable never fell below 6.17 GiB, and
swap stayed at 0.7 MiB. GPU power had a median of **172 W** over the 626 samples with
utilization above 50%, one sample at 184 W, and a peak temperature of 89 °C
([context-262144/memory-summary.json](context-262144/memory-summary.json)).

The needle test ran with `--lengths 32k,128k,256k` and found **all nine needles**.
Actual prompt lengths were 32,034–32,035 tokens for `32k`, 122,798–122,800 for
`128k` and 249,309–249,310 for `256k`
([context-262144/needles.json](context-262144/needles.json)). All three `256k`
cases and the third `32k` and `128k` cases reused 16,384 prompt tokens; the others
reused zero.

## Limitations

This is one machine, one quantization, two context limits and a small synthetic
workload. Long output, sampled decoding, thinking, coding-task correctness, tool
use and a sustained thermal run were not evaluated. A
notebook's power and thermal behaviour depends on its power mode, cooling and
ambient temperature; other power modes were not tested. The README's numbers for
other machines are not a controlled comparison.
