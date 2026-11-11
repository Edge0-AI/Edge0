# Community benchmark on RX 9060 XT with IQ2_XS and IQ3_XXS

Measured October 9–10, 2026 by [isoux35](https://github.com/isoux35). Sharing results from my Windows desktop: Strata 0.1.40.3, Qwen3.8-Flash-Next GSQ-RCO, one 16 GB AMD GPU, and 64 GB RAM.

The goal was to find a practical configuration for local coding and larger context. **IQ2_XS with prefill 8,192** was the configuration I kept. It was faster than IQ3_XXS on the matched synthetic coding tasks, and one 113,501-token recall request completed under a 122,880-token context limit. That recall check is not a demonstration of reliable coding at 120K.

## Hardware and software

| Component | Setup |
| --- | --- |
| GPU | AMD Radeon RX 9060 XT, 16 GB, gfx1200; single GPU |
| CPU | Ryzen 7 9800X3D, 8 cores / 16 threads |
| RAM | 64 GB DDR5-6000; approximately 61.66 GiB usable |
| OS | Windows 11 Home 25H2, build 26200.9457 |
| Driver | 32.0.31041.1004 |
| Strata | Prebuilt Windows HIP engine 0.1.40.3; wrapper checkout `d5ea713` |
| Packaged runtime | ROCm `10.2.0a20260930`; hipBLASLt `100500` per BUILD.json |
| Client | OpenCode 1.18.34 in Ubuntu 24.04 WSL; OpenChamber 1.24.2 |

The engine runs natively on Windows, not Vulkan or WSL. The wrapper checkout is not the binary's verified source commit. Storage model, PCIe link, power limits, and temperatures were not measured. Other desktop applications and client services remained running.

## Model and configuration

Models: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), **IQ2_XS and IQ3_XXS**, using the existing prepared packs and Q2_0 MTP runtime pack. Exact filenames, sizes, available revision metadata, and provenance gaps are in [METHODS.md](METHODS.md#model-artifacts).

Common engine settings, with machine-specific paths omitted:

```text
STRATA_HIP_WMMA=1
--expert-cache auto --prefill 8192
--spec 4 --spec-min-p 0.5
--kv int8 --kv-resident 32768 --vram-reserve-mib 2048
--max-context 122880
```

Prefill trials varied only the prefill size. The coding comparison used **32,768 context for both models**, not 122,880. One parallel request, vision and speed projection off. Sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0, seed 42, repeat penalty 1.0, presence/frequency penalties 0. No new calibration was forced.

## Method and reproduction

- **Prefill:** fresh engine for each of four batch sizes; one excluded warmup, then three 8K and three 28K recall requests. Low reasoning, 1,024-token output cap. Actual input lengths were 8,019 and 28,180 tokens, outputs 111–120 tokens, with zero reused prompt tokens. Expert and OS caches were retained within each arm; these are not fully cold-cache measurements.
- **Coding:** three synthetic TypeScript tasks, three fresh projects/sessions per task and model. Medium reasoning, 8,192 output allowance, Build agent limited to 12 steps and two repair rounds. Only permitted source files and the test command were allowed. Acceptance tests were also checked independently with Node.js v24.21.0.
- **Timing:** loading excluded. Prompt/decode throughput comes from engine timings; decode includes reasoning. Direct wall time includes HTTP and output reception. First streamed token can be reasoning, not answer text. Coding decode is weighted across engine requests and can include client helpers.
- **Memory:** whole-host available physical RAM, sampled every 0.5 seconds for direct requests. GPU figures are whole-adapter counters, not process-isolated allocations.

Configurations ran in fixed order, not randomized: prefill 2,048 → 4,096 → 8,192 → 16,384; coding IQ2 before IQ3. Background work and adaptive expert-cache state can affect the comparison.

[METHODS.md](METHODS.md#reproduction) contains the commands and exact coding prompt. The direct [reproduction script](scripts/reproduce_direct.py) sends requests but does not change server settings or execute generated tools. [summarize.py](scripts/summarize.py) recomputes the published aggregates offline.

## Results

### IQ2_XS prefill

Each cell is median **[min–max]** over three runs. Prompt/decode units are tokens per second; total latency is seconds. All requests had zero prompt-prefix reuse.

| Prefill | Input tokens | Prompt tok/s | Decode tok/s | Total seconds |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 8,019 | 346.3 [344.1–347.2] | 53.2 [49.7–57.7] | 25.400 [25.281–25.765] |
| 4,096 | 8,019 | 593.3 [592.7–597.4] | 52.7 [51.8–55.7] | 15.703 [15.682–15.772] |
| 8,192 | 8,019 | 879.4 [876.3–881.8] | 56.3 [53.1–59.1] | 11.212 [11.111–11.423] |
| 16,384 | 8,019 | 876.3 [869.1–881.6] | 55.3 [53.2–59.8] | 11.384 [11.189–11.428] |
| 2,048 | 28,180 | 347.0 [344.4–351.6] | 57.9 [57.5–59.4] | 83.355 [82.183–84.013] |
| 4,096 | 28,180 | 593.1 [588.2–595.3] | 57.0 [56.8–58.0] | 49.558 [49.393–50.104] |
| 8,192 | 28,180 | 807.4 [806.8–809.1] | 57.6 [56.0–58.0] | 37.008 [36.915–37.021] |
| 16,384 | 28,180 | 811.2 [806.5–811.5] | 57.8 [57.3–58.5] | 36.778 [36.757–37.087] |

Moving from 2,048 to 8,192 cut median total latency by about **56%** at both tested lengths. 16,384 offered no meaningful further gain. Minimum available RAM remained above 8.75 GiB across these measured arms. Per-run records, warmups, and output text: [data](data).

### IQ2_XS versus IQ3_XXS at matched 32K settings

| Measurement | IQ2_XS | IQ3_XXS |
| --- | ---: | ---: |
| Tasks passed | 9/9 | 9/9 |
| Acceptance checks across repeats | 243/243 | 243/243 |
| Median task seconds [min–max] | 99.988 [55.463–145.326] | 136.321 [81.012–166.632] |
| Weighted engine decode tok/s | 49.61 | 42.52 |
| Minimum available host RAM, GiB | 8.92 | 3.11 |
| Peak dedicated GPU counter, GiB | 14.88 | 15.04 |
| Peak shared GPU counter, GiB | 34.16 | 41.12 |

IQ2's overall median task time was about **27% lower**, with about **17% higher weighted decode throughput**. Both passed these tasks; this is not evidence that their general answer quality is equal. The checks are 81 across three fixtures, repeated three times—not 243 distinct tests. Shared GPU memory is system memory, not additional VRAM; dedicated/shared peaks should not be added together.

Per-run results: [coding-per-run.json](data/coding-per-run.json). Original [fixtures](fixtures) and generated [solutions](solutions) are included for independent checking. This is not a comparison against the previous dense Qwen 27B llama.cpp model.

### Large context and follow-up cache

One IQ2 request at context 122,880, prefill 8,192:

| Input / output / reused tokens | Prompt tok/s | Decode tok/s | First streamed token | Total time | Minimum available RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 113,501 / 119 / 0 | 789.9 | 46.9 | 144.157 s | 146.629 s | 9.49 GiB |

All three markers near the start, middle, and end were recalled. **Single run, simple recall only**; output cap was 1,024, and an 8K answer at the context limit was not tested. [Raw record](data/selected-120k-smoke.json).

In a separate OpenCode backend check, the first approximately 22K-input response took 62.468 seconds; three short follow-ups took 1.276 / 1.181 / 1.085 seconds with over 22K tokens reused. All four passed. These are assistant timestamps, excluding polling delay—not a timed OpenChamber UI benchmark. [Cache data](data/cache-opencode.json).

## Correctness and limitations

- Recall content passed **24/24**, but strict plain JSON passed **0/24**: responses used Markdown fences. The near-limit answer had the same formatting issue.
- Three medium-reasoning smoke checks passed: quote arithmetic, validation classification, and one correct tool call. The JSON cases used `response_format`, whose buffering/validation can affect streaming latency; this is not a grammar-decoding guarantee. The tool call was checked, not executed.
- One undelivered coding submission and one read-only check returned proxy HTTP 502. The empty/idle session was verified before sending the undelivered prompt once; completed tasks were not replayed. Harness idle timing and CRLF hash checks also needed correction. Details are retained in [METHODS.md](METHODS.md#incidents-and-stability-limits).
- No relevant Windows crash/display/WHEA events appeared during the approximately 72-minute window. No overnight stability or thermal endurance claim is made.
- Larger real repositories, complex reasoning over 120K history, vision, parallel requests, and performance beyond the tested context were not validated.

**Current choice for this desktop:** IQ2_XS, prefill 8,192, context 122,880, with normal client compaction retained. IQ3_XXS remains a fallback. This report changes no upstream engine code or defaults.

Only sanitized benchmark data and synthetic code are included. Credentials, personal paths, home-network addresses, real projects, chat histories, raw logs, screenshots, and executable hashes are excluded.
