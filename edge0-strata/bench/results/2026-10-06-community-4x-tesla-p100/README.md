# Community benchmark on 4x Tesla P100 16 GB (Pascal, sm_60)

Measured on 2026-10-06 by apollo-mg. This is the experimental CUDA 12 engine on Pascal, which docs/OLDER_GPUS.md lists as "not measured" for the P100.
- **Model and setup:** the installer's configuration for the GSQ-RCO IQ3_XXS model across all four cards.
- **What was measured:** short, ~4.4K and ~36.4K token prompts.
- **Main limitation:** a single machine with the GPU clocks pinned at 1063 MHz, and the engine built with CUDA 12.4, not the 12.8 / 12.9 you test with.

## Hardware and software

- **GPUs:** 4x Tesla P100-PCIE-16GB (compute capability 6.0), 150 W power limit.
  - SM clock during every run was 1063 MHz, from per-second nvidia-smi samples.
- **CPU and RAM:** 2x Intel Xeon E5-2650 v3 (AVX2, no AVX-512); 128 GB DDR4-2133 ECC (121 GB visible).
- **Storage:** Samsung SSD 860 (SATA), about 76 MB/s on large sequential reads during this session. That only affects the first
  load.
- **PCIe:** not measured.
- **OS and driver:** Ubuntu 26.04 LTS (kernel 7.0.0-34-generic), NVIDIA driver 580.173.02.
- **CUDA:** 12.4 (`/usr/bin/nvcc`, V12.4.131), with gcc-13 as the CUDA host compiler. The system default is gcc 15,
  which CUDA 12.4 does not support.
- **Strata:** v0.1.39 (tag at `a1641e9`), source build by setup.
  - There was no ready-made CUDA 12 engine for this setup ("no ready-made engine ... compiling instead").
  - Engine: `engine-cuda12/strata`, built by setup with `STRATA_NVCC=/usr/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-13`,
    `--cuda 12`.
- **Background workloads:** none. The machine was idle apart from the benchmark.

## Model and configuration

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_XXS at the revision setup pins (`ed59f92`).
  - Shards `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (sha256 `219ea929...56d15`) and `-00002`
    (`316b46f3...1e113`), both verified.
  - The files were already on disk and passed with `--gguf-dir`.
- **Vision:** off (`--vision no`).
- **Packs and profiles:** the pack and the MTP draft layer (q2_0 experts) were built by setup, with its default
  expert profile.
- **Context and KV:** max context 65536, `--kv int8`, KV streaming on (`--kv-resident 32768`).
- **Expert cache:** auto, with `--remote-expert-opt` and a layer split across GPUs 0-3 (auto). `--prefill auto`.
- **MTP and sampling:** `--spec 4 --spec-min-p 0.5`; reasoning effort `none`; temperature 0.
- **Not enabled:** calibration and the experimental speed projection.

```text
./setup.sh --setup --family qwen --model IQ3_XXS --gguf-dir <folder with both shards> --gpus 0,1,2,3 --cuda 12 --vision no --yes
engine args (from strata-iq3_xxs.json):
--pack .../packs/iq3_xxs --native ...IQ3_XXS-00001-of-00002.gguf --ple-gguf ...IQ3_XXS-00002-of-00002.gguf
--expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp .../mtp/rt --max-context 65536 --kv int8 --kv-resident 32768 --remote-expert-opt
```

## Method

- **Every request is fresh:** each request starts with a unique nonce line, so no prefix is reused. The engine log
  reports `0 reused` for every request.
- **Request settings:** output capped at 256 tokens; streaming; `reasoning_effort: none` (no reasoning tokens were
  produced).
- **Short prompts:** the two LocalMaxxing canonical prompts (`canonical_reasoning-v1.txt`, `canonical_code-v1.txt`),
  sent by the LocalMaxxing `lmx` v0.1.48 CLI.
  - One untimed warm-up request, then three timed requests, per server start.
  - reasoning-v1 over two server starts, code-v1 over one.
- **Long prompts:** Strata's own docs (`docs/*.md` + `README.md` at the measured commit) plus one question, built by
  `strata_bench.py`.
  - Three runs each, with no separate warm-up; the first ~4.4K run is the first request after load.
- **Timings:**
  - Prompt and decode throughput come from the engine's own `strata serve: prompt ... generated ...` lines, all of
    which are in `runs.json`.
  - TTFT is client-side: from sending the request to the first streamed token, which is answer text.
- **Memory:** a snapshot after load (`free`, nvidia-smi).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| reasoning-v1 (short) | 301-307 | 0 | 256 | 6 | 94.0 (92.9-94.7) | 33.5 (32.0-35.3) | 3.30 (3.27-3.34) |
| code-v1 (short) | 239-242 | 0 | 256 | 3 | 86.1 (86.0-88.9) | 32.6 (32.2-33.2) | 2.83 (2.75-2.86) |
| Strata docs, ~4.4K | 4441 | 0 | 256 | 3 | 239.4 (222.3-240.8) | 29.0 (27.5-29.4) | 18.63 (18.52-20.07) |
| Strata docs, ~36.4K | 36356-36359 | 0 | 233-256 | 3 | 593.5 (583.5-594.9) | 28.0 (27.6-29.4) | 61.59 (61.38-62.58) |

- **MTP drafts accepted / drafted:**
  - short reasoning 167-174 / 215-240;
  - short code 167-168 / 228-238;
  - 4.4K 135-146 / 208-226;
  - 36.4K 131-148 / 205-209.
- **Decode expert cache hit rate:** 98.7-99.7 % over the 12 short-prompt requests. ~0 % of routed experts went over
  PCIe; the whole IQ3_XXS expert arena fits in the four cards.
- **Memory after load:** 49 GB of RAM used (72 GB available); VRAM 13.1 / 13.4 / 15.5 / 15.8 GB on GPUs 0-3.
- **Two 36.4K runs ended before the cap** (233 and 235 tokens; the engine generated fewer than 256, so the model stopped on its own; finish reason not recorded).
- **Failures:** none. Every request completed.

**For context only (a different engine):** the same weights on llama.cpp (a fork at `0b2789f23`, tensor split, the
Flash-Next MTP head at draft 3) decode the two short canonical prompts at a similar speed (33.4 tok/s median,
measured by the same `lmx` CLI).

## Correctness and limitations

- Every answer was coherent text that addressed the prompt. Answer heads are in the per-run data; a short-prompt
  example begins "### Part 1: Budget Calculation".
- No accuracy check, needle test, tool call or image test was run.
- **Untested:**
  - other packs, including IQ3_S and the Unsloth 4-bit;
  - prompts above ~36K;
  - concurrency;
  - unpinned clocks;
  - the CUDA 12.8 / 12.9 toolkits.
- Pascal emulates `__dp4a` (docs/OLDER_GPUS.md). This report does not separate that cost.

Files: `runs.json` (engine timing lines and the table rows), `client_timing_long_prompts.jsonl`, `strata_bench.py`,
and the two canonical prompts.

(Test designed and run with my agent, Claude; I reviewed it before posting.)
