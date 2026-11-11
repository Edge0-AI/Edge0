# Community benchmark on 2x RTX PRO 4500 Blackwell, Swift IQ3_XXS, engine 0.1.40.1

Measured on 2026-10-06 by qni-live. Layer split over two 32 GB cards, Swift 1.5 IQ3_XXS. Two parts: (1) single stream, 1k to 256k prompt tokens, engine 0.1.40.1 against 0.1.39 measured right after it on the same setup; (2) several concurrent requests with and without `--batch` (the all-resident batch fix from 0.1.40, issue #776). One prompt family only (code explanation). Limits are listed at the end.

The tests were run by an AI agent (Claude Code) on my machine, with me deciding the plan and reviewing the results and this report.

## Hardware and software

- GPU: 2x NVIDIA RTX PRO 4500 Blackwell, 32 GB GDDR7 each (GB203, SM 12.0, 200 W limit, unchanged), no NVLink, PCIe Gen 5 x16 to each card (maximum link as reported by `nvidia-smi`)
- CPU: AMD Ryzen Threadripper 7960X, 24 cores / 48 threads. RAM: 64 GB DDR5 (61 GiB usable)
- Storage: model on a 3.7 TB NVMe at PCIe 4.0 x4. The disk only matters at start.
- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic. Driver 595.91.07. CUDA 13.4 toolkit
- Strata: tag v0.1.40.1, commit `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`, **source build** for SM 120 (`build.txt`). Control: v0.1.39, commit `6f32ec070f23ced9f50e704d854d775da52591ab`, source build
- Background workloads: none besides the benchmark. The server was stopped and restarted between configurations.

## Model and configuration

- Model: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_XXS, files `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` (39.8 GB + 36.2 GB). Pack built by setup, shipped expert profile, MTP runtime from `Strata-data/mtp/rt`
- Context 262144, KV `k8v4`, expert cache auto, prefill auto, `--spec 5`, `--spec-min-p 0.5`, layer split 25 over GPUs 0 and 1
- Solo runs: vision encoder loaded (no images sent), `--vram-reserve-mib 700`
- Batch runs: no vision, `--vram-reserve-mib 700`, `--batch 2` or `--batch 4`, each with `--batch-groups 2 --trim-stage-weights`. **No `STRATA_*` environment overrides**; in particular no `STRATA_VERIFY_ALL_RESIDENT=0`
- With 2 slots 100% of the experts are VRAM-resident; with 4 slots 95% (the startup log says so)
- Sampling: server `presence_penalty` set to **0.0** in all runs (our daily config uses 0.2; batch windows apply no penalties, so all arms use the same value). `bench.py`: `temperature 0`. `parlast.py`: `temperature 0.7, top_p 0.95, top_k 20`. Reasoning at the server default. No calibration, no experimental speed projection

Configs as run: `config-solo.json`, `config-batch2.json`, `config-batch4.json`. Start: `serve/server.py --engine strata --config <file> --port 8080`.

## Method

- Single stream: `bench.py` (`temperature 0`, `max_tokens 256`, the reasoning text counts toward the 256). Prompt = Strata source and docs cut to the target length plus an "explain and propose three refactorings" task, with a random nonce first so nothing is reused from the cache (`reused` is 0). One warm-up per length, then 3 measured runs; speeds are read from the server log line. The 256k row is new (target 245,000 tokens)
- Concurrency: `parlast.py`, 8,000-token prompts (a different one per request, nonce first), 400 tokens out, streaming through the server, 1-4 concurrent requests, 3 rounds, median. "Total tok/s" = all generated tokens / wall time of the round, **including** prompt reading (about 2.5 s per request). It is not decode speed. The 32k check used 32,000-token prompts and 300 tokens, 2 rounds.
- Needle recall: `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`, run against the 0.1.40.1 solo server
- Scripts: `run-series.sh` (main series), `run-control-0.1.39.sh` (control). Order: 0.1.40.1 solo, batch 2, batch 4, then 0.1.39 solo

## Results

### Single stream, engine 0.1.40.1 (3 measured runs per length after one warm-up)

| Context | Actual prompt tokens | Reused | Generated | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | Time to first token, s median (range) | Draft accept |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| 1k | 1,023 | 0 | 256 | 3 | 1,889 (1,888-1,897) | 162.3 (158.2-166.7) | 0.5 (0.5-0.5) | 71% |
| 4k | 4,234 | 0 | 256 | 3 | 3,073 (3,068-3,073) | 154.8 (144.7-155.0) | 1.4 (1.4-1.4) | 67% |
| 32k | 30,797 | 0 | 256 | 3 | 5,244 (5,242-5,262) | 142.3 (140.7-152.8) | 5.9 (5.9-5.9) | 68% |
| 128k | 120,608 | 0 | 256 | 3 | 6,127 (6,114-6,143) | 133.6 (125.7-154.4) | 19.7 (19.6-19.7) | 63% |
| 256k | 248,489 | 0 | 256 | 3 | 5,977 (5,976-5,983) | 142.5 (137.7-143.0) | 41.6 (41.5-41.6) | 59% |

Time to first token is derived as actual prompt tokens / prompt tok/s (no separate timer). The 256k prompt had 248,489 tokens; no failure, RAM stable.

### 0.1.40.1 against 0.1.39 (same setup, 0.1.39 measured right after)

| Context | Prompt tok/s 0.1.39 | Prompt tok/s 0.1.40.1 | Change | Decode tok/s 0.1.39 (range) | Decode tok/s 0.1.40.1 (range) | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 1,870 | 1,889 | +1.0% | 150.1 (139.1-166.5) | 162.3 (158.2-166.7) | +8.1% |
| 4k | 3,036 | 3,073 | +1.2% | 138.2 (129.9-139.8) | 154.8 (144.7-155.0) | +12.0% |
| 32k | 5,136 | 5,244 | +2.1% | 143.8 (142.7-166.9) | 142.3 (140.7-152.8) | -1.0% |
| 128k | 6,073 | 6,127 | +0.9% | 119.1 (116.6-121.8) | 133.6 (125.7-154.4) | +12.2% |
| 256k | 5,958 | 5,977 | +0.3% | 135.3 (132.6-146.1) | 142.5 (137.7-143.0) | +5.3% |

Prompt reading is 0.3-2.6% faster. The decode difference is not established: the ranges overlap at 1k, 32k and 256k, and draft acceptance moves the speed by several percent from run to run. I would not call a decode gain certain. These two runs are not comparable with the 0.1.36 numbers I posted earlier (different server run, other sampling settings).

### Concurrent requests (8k prompts, 400 tokens out, total tok/s incl. prompt reading, median and range of 3 rounds)

| Concurrent requests | Queue (no `--batch`) | `--batch 2` | `--batch 4` |
| ---: | --- | --- | --- |
| 1 | 72.5 (67.6-79.6) | 72.6 (70.1-75.6) | 71.7 (66.5-71.7) |
| 2 | 75.1 (73.3-75.5) | 81.3 (79.9-85.5) | 74.2 (73.9-74.7) |
| 3 | not measured | 80.8 (76.9-82.7) | 89.1 (86.5-90.0) |
| 4 | 74.2 (71.4-79.5) | 88.2 (81.1-88.6) | 95.5 (95.4-96.4) |

With two requests: +8% for 2 slots; 4 slots do not help with only 2 requests (as before the fix). With four requests: +19% (2 slots), +29% (4 slots). Two concurrent 32k requests on 2 slots: 36.7 tok/s total (no queue arm measured at 32k). No request failed and the engine did not exit in any round (`data/`).

### Correctness

- Needle recall: **6 of 6 found** (32k and 128k, depths 10/50/90; `needles.json`).
- Tool calls (`tools-and-exactness-batch2.log`, 2 slots): 9 of 9 valid, alone and two at once.
- Same prompt, `temperature 0`, 250 tokens: solo and batch output was identical for 3 of 3 prompts. Three prompts are too few to claim byte-exactness in general.

## Failures and untested

- No failed requests and no engine exits in the runs above. The `--batch` start that failed on 0.1.39 with `verify batch: timed out at layer 0` (issue #776) works on 0.1.40.1 on this all-resident split.
- Not measured: Windows; images with `--batch`; prompts above 32k with several concurrent requests; a queue arm at 32k; `presence_penalty` above 0 and repetition loops in batch mode; `--batch-mtp` (one GPU only in this version).
- Single prompt family. Output speed moves several percent with the generated text. The 0.1.40.1 and 0.1.39 solo numbers are one server start each, not repeated across restarts (we see about 5% spread over restarts).
