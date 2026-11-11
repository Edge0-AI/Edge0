# Community benchmark: RTX 5090, Core Ultra 9 285K

Measured on 2026-09-30 by [hagope](https://github.com/hagope), on a Linux
machine called OMAN. This tests Strata 0.1.29 with the original Flash-Next
IQ2_XS, one GPU, and a 131,072-token context limit.

Median decode throughput was **179.4 tok/s at 4,096 prompt tokens, 175.7 tok/s
at 32,768, and 165.0 tok/s at 128,000**. These are synthetic code-explanation
requests with greedy decoding and a 256-token output cap. They do not establish
general answer quality or performance on other workloads.

## Hardware and software

- NVIDIA GeForce RTX 5090; 32,607 MiB reported VRAM; 575 W power limit.
  PCIe capability: Gen 5, x16. The engine's startup transfer probe reported
  50.0 GB/s host-to-device. GPU clocks were not fixed for this test.
- Intel Core Ultra 9 285K; 24 logical CPUs. The engine selected AVX2 and
  23 expert-pool workers plus its host thread.
- 64 GB installed RAM; Linux reported 62.19 GiB usable. Local NVMe storage:
  WDC WDS200T2B0C-00PXH0. 8 GiB system swap.
- Ubuntu 24.04.4 LTS, kernel 6.8.0-142-generic, NVIDIA driver 595.84.
- Source commit `d6708a4aae15b4860000d54c8af9e84d684bce09`; engine 0.1.29.
  Locally compiled CUDA architecture 120, including the GPU vision helper.
  See [BUILD.json](BUILD.json).
- CUDA 13.0.2 container; NVCC 13.0.88; GCC 13.3.0; Python 3.12.3.
  Base image: `nvidia/cuda@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6`.
  Installed Python dependencies are listed in [python-packages.txt](python-packages.txt).
- The regular inference server, GPU TTS, and GPU embedding service were paused.
  Other CPU services remained running. The GPU was dedicated to Strata and its
  vision helper; this was not a completely isolated operating system.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`
- `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf`
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf`

All three local GGUF SHA-256 hashes matched the revision's published LFS hashes.
[model-provenance.json](model-provenance.json) contains filenames, byte sizes,
revisions, and verified hashes. MTP was fetched by the installer from
`Qwen/Qwen3.8-Flash-Next`; its API reported revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` at verification time. The installer
uses `main` for those tensor-range downloads. Their individual output hashes
are retained in [mtp-manifest.json](mtp-manifest.json); the full BF16 upstream
shards were not downloaded or independently hash-verified.

The unmodified installer prepared the native IQ2_XS pack and Q2_0 MTP experts.
[artifact-hashes.json](artifact-hashes.json) identifies the executables,
default expert profile, tokenizer, pack metadata/dense weights, and draft pack.

- Context 131,072; INT8 KV; 32,768 KV cells per attention layer resident on GPU.
- Expert cache `auto`: 17,463 slots, 23.44 GiB VRAM, profile-prefilled with
  **no eviction**. The bundled expert profile was used without calibration.
- Prefill `auto` selected chunks of 8,192 tokens, borrowing 3,421 expert-cache
  slots during prompt processing.
- MTP `--spec 4 --spec-min-p 0.5`; built-in suffix drafting remained enabled.
- Low-RAM mode off; GPU vision enabled, 1,024 image tokens and a 700 MiB VRAM
  reserve. The vision helper warmed up; benchmark requests contained no images.
- Experimental speed projection off; no custom control vectors or calibration.
- Reasoning disabled, temperature 0, maximum 256 generated tokens per speed run.

The complete measured configuration is [strata-iq2_xs.json](strata-iq2_xs.json).
Startup choices and every request's timing/draft statistics are in
[engine.log](engine.log).

## Method and reproduction

Use the attached [Dockerfile](Dockerfile), the pinned source commit, and an empty
data directory. Mount a working directory as `/workspace`, with the checkout
at `/workspace/Strata`. Run setup inside the CUDA container with GPU access,
host IPC, and unlimited memlock:

```bash
bash ./setup.sh --family qwen --model IQ2_XS --context 131072 --kv int8 \
  --vision gpu --experimental-speed-projection off --gpu 0 --low-ram off \
  --build --yes --no-start --port 18080 --data-dir /workspace/Strata-data
```

Launch the prepared server on loopback inside a container using host networking:

```bash
/workspace/Strata/.venv/bin/python serve/server.py --engine strata \
  --config strata-iq2_xs.json --host 127.0.0.1 --port 18080
```

Run [benchmark.py](benchmark.py) inside that container, after placing the script
at `/workspace/benchmark.py`. It generates deterministic synthetic Python
functions, adds a different nonce near the start of each request, and counts the
complete rendered chat prompt using Strata's tokenizer. It adjusts the filler
to the target token count and verifies the count against the server afterward.
The resulting request hashes are in [results.json](results.json).

```bash
/workspace/Strata/.venv/bin/python /workspace/benchmark.py
/workspace/Strata/.venv/bin/python tools/needle_bench.py \
  --url http://127.0.0.1:18080 --lengths 32k,128k --depths 10,50,90 \
  --out /workspace/results/needles.json
```

One short warm-up request was excluded. Three runs at each length were executed
serially in increasing-length order on the same loaded engine. All nine speed
requests processed their entire prompt: **zero reused tokens**. The GPU expert
cache was filled at startup and retained between requests; no engine restart or
OS page-cache reset separated them. Downloads and file verification had already
warmed the filesystem cache. Loading time is excluded.

The script measures streaming TTFT from immediately before the HTTP request
until the first nonempty text delta, ignoring empty role chunks and keep-alives.
With reasoning off, that delta contains answer text. Total latency ends at
stream completion. Both timings include HTTP and frontend tokenization on OMAN
over loopback. Engine prompt throughput uses freshly read tokens divided by
`prompt_ms`; decode throughput uses `engine_generated / decode_ms`. The JSON
preserves the engine timing values and output text for each run.

## Results

Each cell below is the median **[minimum–maximum]** of three runs. Every request
generated 256 tokens and stopped at the output limit, with no failed or cancelled
speed requests.

| Prompt tokens | Reused | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0 | 4,269.8 [3,510.2–4,273.3] | 179.4 [161.3–184.8] | 0.974 [0.973–1.183] | 2.392 [2.351–2.761] |
| 32,768 | 0 | 5,543.2 [5,527.1–5,549.6] | 175.7 [166.1–181.8] | 5.950 [5.942–5.968] | 7.390 [7.348–7.498] |
| 128,000 | 0 | 5,778.7 [5,767.0–5,791.9] | 165.0 [154.8–169.3] | 22.262 [22.212–22.308] | 23.850 [23.765–23.856] |

Raw per-run records: [results.json](results.json). Calculated aggregates:
[summary.json](summary.json). The lower first 4K run is retained in the range.
Decode expert-cache hit rates were 97.8–99.7% for these nine requests.

[monitor.py](monitor.py) sampled the host once per second, starting before model
load and ending after the recall checks. [telemetry.jsonl](telemetry.jsonl)
contains 250 samples over 255.9 seconds, with the following observed peaks:

- GPU memory: **31,785 MiB (31.04 GiB)**, system-wide on the selected GPU.
- Host RAM: **45.54 GiB**, calculated as `MemTotal - MemAvailable`. This includes
  Strata, other services, and non-reclaimable system memory; it is not process RSS.
- Swap allocation increased from **0.25 MiB to 425.50 MiB**, including increases
  during the speed suite. Swap I/O rates were not measured, so this is not a
  claim that inference ran without paging. No out-of-memory error occurred.

Sampling and the server's own telemetry were active during inference. These
are sampled peaks; brief allocation spikes between samples may be missed.
Aggregates are also recorded in [memory-summary.json](memory-summary.json).

## Recall and limitations

The repository's unchanged `tools/needle_bench.py` found **all six needles**:
depths 10%, 50%, and 90% at both requested lengths. Actual prompt lengths were
30,804–30,805 tokens for `32k` and 126,318–126,320 for `128k`. All answers exactly
matched the expected code words. See [needles.json](needles.json).

The final 128K recall case reused 16,384 prompt tokens; the other five reused
zero. Recall latency is therefore not used in the fresh-prompt speed table.
The script chooses needle depth by character position, not token position.

This is one machine, one quantization, one configuration, and a small synthetic
workload. Long output, sampled decoding, thinking, coding-task correctness,
vision, tool use, multi-request concurrency, and a sustained thermal run were
not evaluated. The built-in needle check measures recall on these six inputs,
not general model accuracy. No same-workload baseline on another engine or GPU
was run; do not interpret the README's different-hardware numbers as a controlled
comparison. The 128,000-token speed prompt also does not fill every token of the
131,072-token context window.
