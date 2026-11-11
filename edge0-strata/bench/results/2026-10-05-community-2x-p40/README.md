# Community benchmark: 2x Tesla P40 22 GB (Pascal, sm_61), 2x Xeon E5-2698 v4

Measured on 2026-10-05 by [sarge18](https://github.com/sarge18) on a Linux server. This tests the original Flash-Next IQ2_XS
with a 32,768-token context on the **experimental CUDA 12 engine for Pascal** (`-DSTRATA_EXPERIMENTAL_SM60=ON`, sm_61), once on a
single P40 and once on **both P40s with the layer split** (config `"gpu": [0, 1]`, which `docs/MULTI_GPU.md` lists as unsupported for
cards below compute capability 7.5). The report was prepared with an AI assistant (Claude Sonnet 5.5, medium effort) under the
maintainer's direction; the full study, which also compares llama.cpp and Ollama on the same weights, is in
[sarge18/p40-llm-engine-bakeoff](https://github.com/sarge18/p40-llm-engine-bakeoff).

Median decode throughput for a short prompt was **19.6 tok/s on one P40 and 34.8 tok/s on both**. Prompts were read at
**348 / 388 tok/s** (4K / 16K prompt tokens) on one card and **353 / 501 tok/s** on both.
Only three runs per cell on a 32K context; the first run of each configuration includes expert-cache warm-up (see Method). The requests are
synthetic and greedy; they do not establish general answer quality or performance on other workloads.

## Hardware and software

- **GPUs:** 2x NVIDIA Tesla P40 (GP102, 22.4 GiB usable of 24 GB, compute capability 6.1), passively cooled, driven by the chassis fans. Power limit left at the
  card default (observed draw up to 278 W); temperatures reached 84 C (one card) and 79 C (both). PCIe 3.0 x16. The engine's PCIe probe measured 11.8-11.9 GB/s host to device for GPU 1
  and 10.3 GB/s for GPU 0 (see the engine logs). GPU 0 sits on NUMA node 0, GPU 1 on node 1; the single-card runs were pinned to node 1 (`numactl --cpunodebind=1 --preferred=1`), the two-card runs were not pinned.
- **CPU and RAM:** 2x Intel Xeon E5-2698 v4 (2x20 cores, 80 threads, AVX2, no AVX-512; the engine used 19 expert-pool workers plus its host thread). 153 GiB RAM (Linux MemTotal 160,445,020 kB). Storage: a
  spinning-disk ZFS RAIDZ2 pool shared with other duties (it affects cold-load times only). 8 GiB swap; the server unit had swap disabled, and the host's swap use stayed flat (about 0.4 GiB, from other processes).
- **OS and runtime:** Ubuntu 26.04, kernel 7.0.0-29-generic, glibc 2.43, NVIDIA driver 580.173.02, CUDA toolkit 12.4 (Ubuntu's `nvidia-cuda-toolkit`) with g++ 13.4.
- **Source:** Strata v0.1.39, commit `6f32ec070f23ced9f50e704d854d775da52591ab` (current `main` at the time of writing), unmodified.
- **Build:** local source build through setup: `CXX=g++-13 CUDAHOSTCXX=g++-13 STRATA_NVCC=/usr/bin/nvcc ./setup.sh --family qwen --model IQ2_XS --context 32768 --vision no --cuda 12 --build --gpu 1 --yes --no-start --no-browser --data-dir <dir> --experimental-speed-projection off`.
  setup compiled the engine for `compute_61/sm_61` with `-DSTRATA_EXPERIMENTAL_SM60=1`. See "Build notes" below for what was needed on Ubuntu 26.04.
- **Other services:** the machine's other GPU service (an Ollama instance) was stopped for each run and restored afterwards, so the cards ran only Strata. Other processes (a file server, an idle virtual machine, small periodic timers)
  kept running, and one service kept about 22 GB of RAM locked in page cache. No long-running scheduled job was due during the runs, judging by the timer schedule (not by logs); small periodic housekeeping timers did run. This was not a fully isolated operating system.
  Each run was started inside a network namespace with loopback only (no Internet), as an unprivileged user with a hard memory limit.

## Model and configuration

**Model:** [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, IQ2_XS (the revision setup pins). Hashes are the Hugging Face LFS SHA-256 values at that revision;
the files were downloaded by setup, which checks only their structure. They were not re-hashed locally, and the files were deleted afterwards:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` 39,225,954,592 bytes `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` 28,800,138,432 bytes `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`

The native pack and the MTP draft layer were prepared by setup (MTP tensors from `Qwen/Qwen3.8-Flash-Next` at the pinned revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`). The expert profile is the bundled `data/expert-profile.bin`.

**Configuration** ([strata-iq2_xs.json](strata-iq2_xs.json) for one card, [strata-iq2_xs-dual.json](strata-iq2_xs-dual.json) for two; `/m` is the model directory, `/work` the checkout):

- context 32,768, `--kv int8`, expert cache `auto`, `--prefill auto`;
- MTP `--spec 4 --spec-min-p 0.5` (native IQ packs require `--spec` of at least 2, so speculation cannot be switched off);
- one card: `"gpu": 1`; two cards: `"gpu": [0, 1]`, `"layer_split": "auto"`. The engine log records "layer split across 2 GPUs: CUDA0, then CUDA1 (split auto)" and chose K=24 (layers 0-23 on the first card, 24-47 and the head on the second), with "the caches hold 24576 of 24576 profiled pairs (~100.0% of the routed mass)";
- no vision, no calibration, no speed projection, no control vectors; reasoning off (`reasoning_effort: none` and `chat_template_kwargs.enable_thinking=false`); temperature 0, `top_k 1`, seed 1.

```text
# server, inside the benchmark's network namespace (cwd /work); one card:
python serve/server.py --engine strata --config strata-iq2_xs.json --port 8080
# both cards: --config strata-iq2_xs-dual.json
```

## Method

[scripts/bench.py](scripts/bench.py) (stdlib Python; the same file as in the study repository) sends OpenAI-style streaming `/v1/chat/completions` requests. The speed tasks are `speed-short` (a 34-token prompt, 400 tokens requested; it generated exactly 400) and
`speed-prefill-4k` / `speed-prefill-16k` (synthetic survey-log filler of 4,000 / 15,920 tokens, 64 tokens requested). Prompts are regenerated deterministically by the script from a fixed seed, so no prompt files are needed. Each configuration
was started cold, then the three tasks ran in order, three times (`--reps 3`), on one server process, without restarting. The prompts are identical across repetitions, so the engine's conversation cache could have reused tokens; the engine timing values report `cache_n = 0` for every run (nothing reused), which is also what the engine log shows ("0 reused + N read").

- **Prompt tok/s** and **decode tok/s** are the engine's own timing values returned in the response (`timings.prompt_per_second`, `timings.predicted_per_second`). Decode counts generated tokens only, not total latency.
- **TTFT** and **total latency** are measured by the client from sending the request to the first generated token / the end of the stream; they include the engine's prompt processing and any queueing (none: one request at a time). The first token is answer text (thinking off).
- **Warm-up:** the first repetition of each configuration is slower (the expert cache fills): see the per-run table. Medians are of all three runs, with the range in brackets. Model load time was not included in any speed number.
- **Memory:** the server unit's cgroup memory peak (anonymous memory plus page cache charged to it) was 34.8 GB (one card) and 35.8 GB (both cards); swap 0. Peak VRAM during the runs (nvidia-smi, 5-second samples) was 22,601 MiB on GPU 1 for one card, and 20,487 MiB (GPU 0) and 22,377 MiB (GPU 1) with both. No paging or out-of-memory event.
  `data/engine-logs/strata-iq2_xs.log` and `strata-iq2_xs-dual.log` are the engine's cumulative logs for every run on that install (they also contain a smoke test and the quality runs). Peaks are over the whole run; the telemetry CSVs (per 5 s: host available memory, swap, per-GPU utilisation, memory, temperature, power) are in `data/telemetry/`.

## Results

Medians over 3 runs; range in brackets. TTFT and total latency are client-side.

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median (range) | Decode tok/s median (range) | TTFT seconds median (range) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| single P40 (GPU1), 34-token prompt, 400 generated | 34 | 0 | 400 | 3 | 53 (16–53) | 19.6 (11.9–20.1) | 0.69 (0.68–2.30) |
| single P40 (GPU1), 4,000-token prompt, 64 generated | 4000 | 0 | 64 | 3 | 348 (124–349) | 16.1 (10.7–20.0) | 11.54 (11.54–32.24) |
| single P40 (GPU1), 15,920-token prompt, 64 generated | 15920 | 0 | 64 | 3 | 388 (314–389) | 23.7 (15.5–35.6) | 41.11 (41.10–50.75) |
| both P40s, 34-token prompt, 400 generated | 34 | 0 | 400 | 3 | 55 (15–55) | 34.8 (10.4–34.9) | 0.66 (0.66–2.39) |
| both P40s, 4,000-token prompt, 64 generated | 4000 | 0 | 64 | 3 | 353 (120–353) | 37.8 (11.9–37.9) | 11.40 (11.40–33.34) |
| both P40s, 15,920-token prompt, 64 generated | 15920 | 0 | 64 | 3 | 501 (376–501) | 37.2 (13.4–37.2) | 31.93 (31.90–42.47) |

Per-run values (run 1 / run 2 / run 3):

| Configuration | Task | Prompt tok/s | Decode tok/s | Drafts accepted / drafted |
| --- | --- | --- | --- | --- |
| single P40 (GPU1) | speed-short | 16 / 53 / 53 | 11.9 / 19.6 / 20.1 | 226 of 369 / 221 of 379 / 210 of 379 |
| single P40 (GPU1) | speed-prefill-4k | 124 / 348 / 349 | 10.7 / 20.0 / 16.1 | 30 of 47 / 31 of 47 / 34 of 55 |
| single P40 (GPU1) | speed-prefill-16k | 314 / 388 / 389 | 15.5 / 35.6 / 23.7 | 35 of 46 / 35 of 46 / 37 of 48 |
| both P40s | speed-short | 15 / 55 / 55 | 10.4 / 34.8 / 34.9 | 208 of 390 / 208 of 390 / 208 of 390 |
| both P40s | speed-prefill-4k | 120 / 353 / 353 | 11.9 / 37.8 / 37.9 | 35 of 48 / 35 of 48 / 35 of 48 |
| both P40s | speed-prefill-16k | 376 / 501 / 501 | 13.4 / 37.2 / 37.2 | 37 of 49 / 37 of 49 / 37 of 49 |

All 9 requests per configuration finished with `finish_reason` `length` (the requested cap was reached, as expected for fixed-length speed requests); none failed or was cancelled. Not measured: 32K and 128K prompts (the context here was 32,768), vision, other quantisations, concurrent requests, and cold model load (a first-ever cold load from disk took 254 s in a separate smoke test; warm loads were 33-76 s).
Full per-run records are in [results.json](results.json) and, with the raw outputs, in `data/` (`*.jsonl`); summary statistics in [summary.json](summary.json).

## Correctness and limitations

A 69-task correctness suite ran twice on the single-card configuration (same prompts, greedy, thinking off): reasoning with computed answers 36 of 40 / 37 of 40 (40 tasks), code tasks graded by executing hidden tests 11 of 12 / 11 of 12 (12), tool calls 8 of 8 / 8 of 8 (8; two are restraint cases that must not call a tool),
completeness checks 4 of 6 / 5 of 6 (6), and a long-context needle recall of three synthetic "access codes" per prompt (longctx-4k: 4124/4124 tokens, recall 100%/100%; longctx-12k: 12056/12056 tokens, recall 100%/100%; longctx-24k: 23952/23952 tokens, recall 100%/100%). The failures read against the raw outputs were model errors (date arithmetic, word reversal, an inverted Roman-numeral loop, a prime list that stopped at 96 entries, `#` headings where `##` was asked).
The task definitions and graders are in [scripts/bench.py](scripts/bench.py) (`python scripts/bench.py --selftest` checks that the graders pass reference answers and fail garbage); the suite has a ceiling effect and cannot rank models. This is not `tools/needle_bench.py`.

**Run-to-run repeatability:** with temperature 0 and identical prompts, 49 of 69 outputs were byte-identical between the two runs (and 2 items changed pass/fail), i.e. generation was not exactly repeatable on this configuration. For comparison, two runs of one Ollama configuration on the same machine (a different model) produced identical outputs; see the study repository.

Limitations: one machine, one day, a synthetic workload; the first run of each cell includes warm-up; the two-card runs were not NUMA-pinned while the one-card runs were; GPU 0 and GPU 1 are not equally close to the CPU that holds the pinned workers.

## Related reports

- #875 (a P40 + RTX 3070 pair on v0.1.39, IQ3_XXS, P40 power-capped at 130 W, DDR4-2400, PCIe 3.0 x8) and #395 (a P40 on IQ3_S, engine 0.1.30) are other Pascal measurements. The setups differ (quantisation, power cap, memory and PCIe), so the numbers are not directly comparable;
  in this report the decode rate on a P40 varies between about 10 and 36 tok/s across the short generations (warm-up and draft acceptance), which is why every run is listed.
- The wider study (llama.cpp and Ollama on the same weights, a 27B control model, accuracy and repeatability) is in [sarge18/p40-llm-engine-bakeoff](https://github.com/sarge18/p40-llm-engine-bakeoff).

## Build notes for Ubuntu 26.04 (glibc 2.43) and Pascal

- CUDA 12.4 (Ubuntu's package) with **g++ 13** built the engine; `docs/TROUBLESHOOTING.md` suggests g++ 14 for newer GCC, but CUDA 12.4 rejects anything newer than GCC 13 ("gcc versions later than 13 are not supported"), so g++-13 was needed with this toolkit.
- A small `.cu` file using `rsqrtf`, `cospif` and `sinpif` compiled cleanly with `nvcc -ccbin=g++-13 -arch=sm_61`, i.e. the glibc 2.43 header clash of #601 (CUDA 12.9) did not occur with 12.4.
- NVIDIA's apt repository for Ubuntu 26.04 carried only CUDA 13.x when checked on 2026-10-05, and CUDA 13 cannot target Pascal, so a newer 12.x would have to come from the runfile.
