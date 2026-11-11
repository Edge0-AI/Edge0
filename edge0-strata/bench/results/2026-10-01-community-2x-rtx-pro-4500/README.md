# Community benchmark: 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS

Measured 2026-10-01 on a Threadripper workstation with two 32 GB Blackwell workstation cards (SM 12.0), engine 0.1.30.
This is **not** the Flash-Next IQ2_XS pack from the first community entry: it is the Swift 1.5 variant at IQ3_XXS,
split over two GPUs. Numbers are not directly comparable to single-GPU entries.

## Hardware

| | |
| --- | --- |
| GPU | 2x NVIDIA RTX PRO 4500 Blackwell, 32 GB GDDR7 each (GB203, 200 W limit), no NVLink |
| PCIe | Gen 5 x16 to each card (read from `nvidia-smi` under load) |
| CPU | AMD Ryzen Threadripper 7960X, 24 cores / 48 threads |
| RAM | 64 GB DDR5 (61 GiB usable) |
| Board | Gigabyte TRX50 AERO D |
| Storage | Model on a 3.7 TB NVMe at PCIe 4.0 x4 (`/mnt/models`). Disk only matters at startup: during generation the server read 23 MiB for 400 tokens (experts are resident in RAM/VRAM, 100% expert cache hits). Cold start was not measured. |

## Software

- Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic
- NVIDIA driver 595.84, CUDA 13.3 toolkit libraries (`/usr/local/cuda-13.3`)
- Strata engine 0.1.30 (release download, not a git checkout; no commit hash)
- Served through `serve/server.py` on port 8080

## Model

- Repo: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
- Files: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (39.8 GB) (2-GPU split) and `...-00002-of-00002.gguf` (36.2 GB)
- Pack built by setup as `swift-iq3_xxs`, shipped expert profile, MTP runtime from `Strata-data/mtp/rt`
- Vision is loaded (`--vision`) but no images are sent

## Launch command (engine, as started by `serve/server.py`)

```
engine/strata --serve --pack Strata-data/packs/swift-iq3_xxs \
  --native Swift-...-IQ3_XXS-00001-of-00002.gguf --ple-gguf Swift-...-IQ3_XXS-00001-of-00002.gguf \
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 5 --spec-min-p 0.5 --mtp Strata-data/mtp/rt \
  --max-context 262144 --kv k8v4 --vision --vram-reserve-mib 700 --layer-split 25
```

The full serve settings file is `settings.json` in this folder (paths are ours, no credentials).

`--spec 5` is not the default (4). We swept N = 3, 4, 5, 6 first (see below) and kept 5. `--layer-split 25` (the
2-GPU split) and `--kv k8v4` are the values already in our running config; we did not tune them here.

## Method

`bench.py` in this folder, against the running server:

- chat completions, `temperature 0`, `max_tokens 256`, thinking left at the server default (the reasoning text counts
  toward the 256 tokens)
- prompt = Strata's own source (2-GPU split) and docs, cut to the target length, plus a "explain and propose three refactorings" task
- a random nonce at the start of every prompt, so no prompt is reused from the cache (`reused` is 0 in `runs.json`)
- one warm-up run per length, then 3 measured runs
- speeds are read from the server log line (`prompt ... read in ... ms`, `... generated in ... ms`)

## Results (median, range of 3 runs)

| Length | Prompt tokens | Prompt tok/s | Output tok/s | Draft accept |
| --- | ---: | ---: | ---: | ---: |
| 1k | 1,023 | 903 (898-908) | 124.0 (122.1-124.5) | 69% |
| 4k | 4,233 | 1,428 (1,428-1,429) | 115.4 (113.8-119.4) | 70% |
| 32k | 30,793 | 2,454 (2,449-2,463) | 120.0 (110.6-120.7) | 68% |
| 128k | 120,605 | 2,785 (2,784-2,794) | 93.0 (90.2-98.4) | 55% |

Needle recall (`tools/needle_bench.py`, 32K (2-GPU split) and 128K at depths 10/50/90): **6 of 6 found** (`needles.json`).

## Draft length sweep (`--spec`)

Same bench idea, one server run per N, 3 runs per cell, output tok/s median:

| `--spec` | short code prompt | 4K | 32K |
| ---: | ---: | ---: | ---: |
| 3 | 114.8 | 112.1 | 117.2 |
| 4 | 122.2 | 110.7 | 118.6 |
| 5 | 132.4 | 113.9 | 121.3 |
| 6 | 116.2 | 112.1 | 121.0 |

The spread between runs is 3-5%, so N = 5 is at most a small gain (about +8% on the short prompt, +2% at 32K).
N = 6 lowers the accepted-draft share (67% against 74-76%). `sweep.py` (2-GPU split) and its raw output are in `spec-sweep/`.

## Limits

- Output speed moves several percent with the generated text; one prompt family only (code explanation)
- Single stream only, no concurrency test
- No comparison run against engine 0.1.32 yet
