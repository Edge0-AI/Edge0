# Community benchmark: RTX 5080 in Docker Desktop on Windows (WSL2)

Measured on 2026-10-07 by [xWinIcex](https://github.com/xWinIcex), on a Windows 10 host
using the repository's own Dockerfile and entrypoint — no bare-metal Linux, no
`setup.sh`, no hand-written launch command. Coder IQ1_M, one RTX 5080, two context
limits (131,072 and 65,536) on the same loaded model.

Median decode throughput was **75.7–80.8 tok/s at 4,096–32,768 prompt tokens and 72.2
tok/s at 130,560** with the 131,072-token limit, and **71.5–78.5 tok/s at 4,096–32,768
and 73.8 at 65,024** with the 65,536-token limit. These are synthetic code-explanation
requests with greedy decoding and a 256-token output cap; they do not establish general
answer quality. The main limitation is the host: everything ran inside Docker Desktop's
WSL2 VM on Windows, which changes both the memory the installer sees and the storage path.

This is, as far as we could tell from the existing reports, the first report from that
environment, so three things that are specific to it are written up in
[Notes for users of this environment](#notes-for-users-of-this-environment).

## Hardware and software

- NVIDIA GeForce RTX 5080; 16,303 MiB reported VRAM; power limit 360 W (maximum 380 W).
  PCIe link **Gen 5, x8**, while the card reports a maximum width of x16. The engine's own
  startup probe measured **28.9 GB/s host-to-device** (`pcie_frac 0.55`, the default).
  Clocks were not fixed and no thermal soak was run; the highest GPU temperature observed
  was 68 °C and the highest power 273 W.
- AMD Ryzen 9 9950X, 16 cores / 32 threads, AVX-512. The engine chose the AVX-512 path,
  48 pool tasks per phase, 15 expert-pool workers plus the host thread on core 0.
- 64 GB installed RAM; the container sees **54.9 GiB** (see the memory note below).
  Storage: `C:` XPG MARS 980 PRO 1908 GB NVMe SSD — the WSL2 VHDX, and therefore the
  image, the model files and `/data` all live there. The machine's other disk is a
  3,726 GB HDD and was not used.
- Windows 10 Pro 26H2 build 26300; WSL 2.7.10.0, kernel 6.18.33.2-2; Docker Desktop
  4.84.0 (client and server 29.6.2, buildx v0.35.0-desktop.2); NVIDIA driver 617.14.
- Strata commit `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` ("Version 0.1.40.3"); engine
  **0.1.40.3**, `source=local`, CUDA architectures `[120]`, vision compiled for CUDA —
  [BUILD.json](BUILD.json).
- CUDA 13.0.48 (`nvcc` V13.0.48); GCC 13.3.0; Python 3.12.3. Installed Python packages:
  [python-packages.txt](python-packages.txt).
- The whole environment is in [environment.txt](environment.txt), including the exact
  `docker run` lines, the WSL2 memory setting and the base image digest.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` at the revision the installer
pins, `5348543e0147355ac9cbcb031184a3546350988e` (2026-09-29):

- `IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` — 29,608,446,496 B
- `IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` — 28,800,138,432 B
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf` — 907,543,008 B

All three local SHA-256 hashes matched the repository's published LFS oids
([sha256.txt](sha256.txt), [model-provenance.json](model-provenance.json)). The MTP draft
layer was fetched from `Qwen/Qwen3.8-Flash-Next` by the installer (31 tensors, 5.214 GB,
per-tensor hashes in [mtp-manifest.json](mtp-manifest.json)); the full BF16 shards were not
downloaded. The unmodified installer prepared the pack (`/data/packs/coder-iq1_m`, 1.4 GB
on disk) and the Q2_0 MTP experts.

Both configurations were produced by the image's own entrypoint from the same
`strata-data` volume; only `CONTEXT` and `REINSTALL` differ between them. The recorded
engine configuration is [config-131072.json](config-131072.json) and
[config-65536.json](config-65536.json):

```text
# 131,072-token limit
docker run -d --name strata --gpus all --network strata-net \
  -p 127.0.0.1:8080:8080 --ulimit memlock=-1 --stop-timeout 60 \
  -v strata-data:/data \
  -e FAMILY=coder -e MODEL=IQ1_M -e CONTEXT=131072 -e VISION=cpu -e GPU=0 strata

# 65,536-token limit, same volume
  ... -e CONTEXT=65536 -e VISION=cpu -e GPU=0 -e REINSTALL=1 strata
```

Settings both configurations share: `--expert-cache auto`, `--prefill auto`, `--kv int8`,
`--spec 4 --spec-min-p 0.5`, MTP on, `--vram-reserve-mib 700`, low-RAM mode off, the
experimental speed projection off, no calibration, no custom expert profile, temperature 0,
`reasoning_effort: none`, 256 generated tokens per speed run. Vision is on with the encoder
on the **CPU** (`VISION=cpu`, 16 threads, 300 image tokens), which is the first thing worth
noting: nothing about this configuration was tuned by hand.

What differs, from the engine's startup log:

| | 131,072 limit | 65,536 limit |
| --- | --- | --- |
| Expert cache | 3,554 slots, 6.78 GiB VRAM | 4,060 slots, 7.73 GiB VRAM |
| Free VRAM at load | 152 MiB | 178 MiB |
| KV | INT8 | INT8, streaming off, resident in VRAM |
| Prefill chunk | 8,192 tokens, borrows 2,172 cache slots | 8,192 tokens |

At 131,072 the engine warned about its own configuration:

```text
strata serve: 152 MiB of VRAM free with everything loaded - LOW: requests may stall;
add --vram-reserve-mib 1060 to the config's args (or lower --max-context)
```

So a 16 GB card at a 128K context leaves almost nothing spare, and the engine says so
itself. Nothing below failed because of it: no request was refused for VRAM and none
stalled, at either limit, including the 130,560-token prompt.

## Method

The benchmark is the script from
[bench/results/2026-09-30-community-rtx-5090/benchmark.py](../2026-09-30-community-rtx-5090/benchmark.py),
attached here as [benchmark.py](benchmark.py), **unchanged except for four defaults**
(`--root /opt/strata`, `--pack /data/packs/coder-iq1_m`, `--url http://127.0.0.1:8080`,
`--out /data/bench`), so the numbers stay comparable with that report. It builds
deterministic synthetic Python functions, puts a different nonce near the start of every
request so no prompt prefix can be reused, counts the complete rendered chat prompt with
Strata's own tokenizer, and verifies its count against the engine afterwards.

One warm-up request was excluded. Three runs were executed at each length, serially, in
increasing-length order, on the same loaded engine. Every measured request read its entire
prompt: **zero reused tokens** in all 33 runs (`reused: 0`, `prompt_read == prompt_total`).
Loading time is not included. Both context limits were measured in their own session on a
freshly loaded model; see the limitations.

Streaming TTFT is measured from immediately before the HTTP request until the first
non-empty text delta, ignoring empty role chunks; with reasoning off that delta is answer
text. Total latency ends at the end of the stream; both include HTTP and front-end
tokenization on loopback. Prompt throughput is fresh tokens divided by the engine's
`prompt_ms`; decode throughput is `engine_generated / decode_ms`. Per-run records — the
engine's timings and draft statistics, the client's TTFT and total latency, the output
text and a SHA-256 of each request body — are in `data-131072/results.json` and
`data-65536/results.json`. The request bodies themselves are not committed because
`benchmark.py` rebuilds them deterministically; compare the recorded `request_sha256`
after a re-run to confirm it reproduced the same prompt.

Memory was sampled once per second from the engine's own `/metrics` by
[monitor.py](monitor.py) and summarised by
[summarize-telemetry.py](summarize-telemetry.py). Every memory number in this report comes
from the engine's view, not from `nvidia-smi`, which inside a container reports the whole
GPU and not the container's share.

## Results

Each cell is the median **[minimum–maximum]** of three runs. Every request generated 256
tokens and stopped at the output limit; there were no failed or cancelled speed requests.

**Context limit 131,072** (`data-131072/`, session of 2026-10-07 ~22:22):

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 2,252.8 [2,138.8–2,267.5] | 75.7 [72.6–77.9] | 1.849 [1.839–1.948] | 5.306 [5.100–5.350] |
| 8,192 | 0 | 3,009.2 [2,998.9–3,063.1] | 79.9 [78.9–80.8] | 2.772 [2.706–2.780] | 5.929 [5.913–5.940] |
| 16,384 | 0 | 3,197.3 [3,184.2–3,304.9] | 79.1 [77.6–81.0] | 5.166 [5.002–5.189] | 8.402 [8.136–8.442] |
| 32,768 | 0 | 3,304.3 [3,271.8–3,311.9] | 80.8 [75.6–82.4] | 9.981 [9.960–10.081] | 13.223 [13.057–13.316] |
| 65,536 | 0 | 3,299.4 [3,269.5–3,310.9] | 73.8 [73.8–77.5] | 19.958 [19.884–20.137] | 23.400 [23.166–23.584] |
| 130,560 | 0 | 3,076.5 [2,942.9–3,200.6] | 72.2 [71.0–76.1] | 42.612 [40.959–44.532] | 46.131 [44.292–48.114] |

**Context limit 65,536** (`data-65536/`, session of 2026-10-07 ~22:37):

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 2,194.1 [2,113.8–2,221.6] | 73.1 [72.1–78.2] | 1.904 [1.882–1.970] | 5.379 [5.127–5.495] |
| 8,192 | 0 | 2,781.6 [2,760.3–2,804.9] | 76.4 [75.2–76.8] | 2.981 [2.963–3.010] | 6.332 [6.288–6.336] |
| 16,384 | 0 | 3,001.3 [2,997.2–3,016.8] | 78.5 [68.4–78.6] | 5.523 [5.480–5.539] | 8.741 [8.707–9.241] |
| 32,768 | 0 | 3,034.9 [3,016.7–3,084.7] | 71.5 [70.3–73.6] | 10.873 [10.702–10.945] | 14.419 [14.148–14.550] |
| 65,024 | 0 | 3,027.7 [3,003.6–3,197.0] | 73.8 [73.7–79.9] | 21.582 [20.441–21.762] | 25.028 [23.619–25.203] |

The two limits are **not** in the expected order: at every shared prompt length the
131,072-token configuration measured equal or slightly faster (decode 80.8 vs 71.5 tok/s at
32,768; prefill 3,304 vs 3,035 tok/s), even though it has the *smaller* expert cache
(3,554 vs 4,060 experts, 6.78 vs 7.73 GiB of VRAM) and only 152 MiB of free VRAM. Decode
ranges at 32,768 do not overlap (128K: 75.6–82.4; 64K: 70.3–73.6), so this is not obviously
run-to-run noise, but with one session per configuration and no interleaving we did **not**
isolate the cause and do not claim that a smaller context is slower in general. It is
reported because it is what was measured, and because the cache sizes alone do not predict it.

Decode expert-cache hit rates were 81.0–87.0% across the 33 measured runs (`hit_rate` in
`results.json`); draft acceptance was 140–162 of the 196–237 drafts offered.

### Memory

From the one-second samples (peaks and ranges; a sampled peak can miss a shorter spike):

| | 131,072 limit | 65,536 limit |
| --- | ---: | ---: |
| GPU memory used, peak (of 15.92 GiB) | 15.82 GiB | 15.74 GiB |
| Host RAM used, peak (of 54.92 GiB) | 29.77 GiB | 29.49 GiB |
| Free VRAM as the engine reports it | 152 MiB | 178 MiB |
| GPU temperature, peak | 68 °C | 65 °C |
| GPU power, peak (limit 360 W) | 267 W | 268 W |
| Disk read, peak | 230 MB/s (recall run) / 3.7 MB/s (speed run) | 3.6 MB/s |

`gpu_mem_used` comes from the engine's `/metrics`, so it is the process-plus-context
allocation rather than the card's total. The 230 MB/s peak was recorded during the 128K
recall run: on very long prompts the experts that the prompt path's borrowed cache slots do
not cover are read from the SSD, which is the behaviour the engine documents.

## Correctness

The repository's unchanged `tools/needle_bench.py` found **all nine needles** on the
131,072-token configuration — depths 10%, 50% and 90% at lengths 32k, 64k and 128k. Actual
prompt lengths were 32,343–32,344, 62,625–62,627 and 125,919–125,920 tokens; every answer
matched the expected code word. On the 65,536-token configuration the same command found
**6 of 6** (32k and 64k; 128k skipped because the server's context is 65,536), and the skip
is recorded in the output rather than silently dropped. See
[data-131072/needles.json](data-131072/needles.json) and
[data-65536/needles.json](data-65536/needles.json).

The vision path was checked with [image-check.py](image-check.py), which draws a code word
with Pillow and sends it as an OpenAI `image_url` data URL. With the encoder on the CPU the
model answered `K7QX-4291` for the drawn `K7QX-4291` in 1.1 s at 261 prompt tokens
([data-131072/image-check.json](data-131072/image-check.json),
[image-check.png](data-131072/image-check.png)). The encoder falling back to the CPU is
visible in the log as `ggml_cuda_init: failed to initialize CUDA` followed by
`strata-vision: on the CPU, 16 threads`, which is what `VISION=cpu` asks for.

## Notes for users of this environment

1. **Docker Desktop's default RAM makes the installer choose the low-RAM mode.** With no
   `%UserProfile%\.wslconfig`, Docker Desktop caps the WSL2 VM at 50% of host RAM, so the
   container saw ~31 GB of the host's 64 GB. `setup.py` reads RAM from `/proc/meminfo`, so it
   did not error — it selected the low-RAM mode ("the GPU holds ~38% of IQ1_M's experts, the
   other ~15 GB stay in RAM, read once from a copy in the model folder"). A 32 GB PC is a
   documented low-RAM case, but on a 64 GB PC it is silent and unwanted. Adding
   `[wsl2]` / `memory=56GB` ([wslconfig.txt](wslconfig.txt)) and restarting
   (`wsl --shutdown`) gives the container 54.9 GiB and the normal mode. Both measured runs
   used the normal mode.
2. **A request needs 8 tokens of headroom.** `prompt + max_tokens + 8 ≤ max_context`, and
   the engine does not truncate. A 65,280-token prompt with `max_tokens: 256` on the
   65,536-token limit was refused with HTTP 400: *"prompt (65280 tokens) + max tokens (256)
   exceeds the context (65536); requests are never truncated. Send a smaller max_tokens (at
   most 248 here)"*. [probe-request.py](probe-request.py) reproduces it. The largest prompt
   measured here therefore leaves 512 tokens: 130,560 of 131,072 and 65,024 of 65,536.
3. **The port is published on `127.0.0.1` only**, so no API key was set
   (`"api_key": false` in `/health`). Other containers reach the same server over a
   user-defined Docker network at `http://strata:8080`; nothing was exposed to the LAN.

## Limitations

- **One session per configuration.** The warm-up state differs between them, and the
  128K-vs-64K comparison in the results is a comparison of two sessions, not of two
  interleaved settings. The engine's own log shows CUDA-graph capture
  (`strata verify: capturing the N-token window`) happening during the first requests of a
  session, which is one candidate explanation for the ordering; it was not tested. Treat the
  difference as an observation to reproduce, not a result.
- **One machine, one quantization, one GPU, one synthetic workload.** Long output, sampled
  decoding, thinking, multi-request concurrency (`parallel`), tool use and a sustained
  thermal run were not evaluated. The needle and image checks measure recall and reading on
  those specific inputs, not general model quality.
- **The PCIe link ran at x8 of a possible x16**, and the card reports a maximum width of
  x16. The engine measured 28.9 GB/s host-to-device; the RTX 5090 report on a Gen5 x16 link
  measured 50.0 GB/s. Whether the narrower link costs anything here was not tested — the
  link is a property of this machine's slot wiring, so absolute numbers may move on a host
  that runs the card at x16.
- **Windows and WSL2 are between Strata and the hardware.** File I/O and page-locked memory
  go through the WSL2 VM; `--ulimit memlock=-1` was passed and the engine reported
  `MAP_HUGETLB unavailable` with transparent huge pages requested instead. Numbers from a
  bare-metal Linux host with the same card are not guaranteed to match.
- **Timings include the whole client path** (HTTP, front-end tokenization, loopback) and are
  measured at the client, not inside the engine.
- The 131,072-token configuration logged the engine's own VRAM warning quoted above. It did
  not cause a failure in any measured run, but it is a real margin of 152 MiB and a reason
  to prefer `--vram-reserve-mib 1060` (or a smaller context) for unattended use.
- One telemetry file, `data-131072/telemetry-speed-overlapped.jsonl`, was written by two
  samplers that overlapped for about eleven minutes (819 of 820 lines parse; the timestamps
  contain one discontinuity). Only peaks and ranges are taken from it. The clean runs are
  `data-131072/telemetry-checks.jsonl` and `data-65536/telemetry-speed.jsonl`.

## Files

| File | What it is |
| --- | --- |
| `benchmark.py`, `monitor.py`, `summarize-telemetry.py`, `image-check.py`, `probe-request.py` | the scripts, runnable as documented above |
| `data-131072/`, `data-65536/` | `results.json` (per-run engine timings, draft statistics, output text, request hashes) and `summary.json`, the telemetry and its memory summary, `needles.json`, the image check, and the engine's status at startup |
| `config-131072.json`, `config-65536.json`, `engine-131072.log`, `engine-65536.log` | the recorded configuration and the unabridged engine log of each session |
| `environment.txt`, `python-packages.txt`, `wslconfig.txt` | host and container environment |
| `model-provenance.json`, `sha256.txt`, `mtp-manifest.json`, `mtp-inventory.md` | what was downloaded and its hashes |
| `BUILD.json` | the engine the image compiled |
