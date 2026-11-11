# Community benchmark: EVO-X2, Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB UMA

Measured on 2026-10-09 by [libratechw](https://github.com/libratechw), using unmodified Strata v0.1.41 (`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`) and Unsloth UD-IQ4_XS. This adds a same-day default/fast long-prompt comparison, two repetitions of the existing 69-task correctness suite, the official needle probe, and parallel-request checks relevant to [#1356](https://github.com/Niko1221/Strata/issues/1356).

The default path scored **65/69 in both passes**, with identical generated content on all 69 tasks. At 128,000 prompt tokens, the fast configuration measured **1,108.6 prompt tok/s and 47.5 decode tok/s** (three-run medians), against 617.4 and 42.1 on the default configuration.

These are small synthetic checks. The correctness score is neither HumanEval nor an agent benchmark, and the fast arm was measured for speed only.

## Relation to other Strix Halo reports

[2026-10-05-community-gfx1151](../2026-10-05-community-gfx1151/README.md) (#917) used the same CPU/GPU class on 0.1.39 with unmerged PRs, to isolate the hipBLASLt tuning table (250.9 to 528.9 prompt tok/s), dense MMQ and the shared-expert stream. This report starts after those landed: the gfx1151/100401 table is bundled with 0.1.41 and used in every arm here, so it is not re-measured. It adds the default versus documented fast configuration on 0.1.41, the 69-task correctness suite, recall up to the native window, and the parallel checks.

The maintainers' [STRIX_HALO.md section 6](../../../docs/STRIX_HALO.md#6-measured) lists UD-IQ4_XS at 128K as 1,320 prompt tok/s and 51.4 output tok/s for the fast configuration (their prompts and method), and 1,299 and 50.7 re-measured on the merged 0.1.40 code with no environment switches at all. The fast arm here sets all nine section 5 switches and measured 1,108.6 and 47.5; the default arm measured 617.4 and 42.1. This report did not isolate the cause of the gaps. Differences that could contribute include the prompts, the engine version, this report's memory plan (the fast arm lends 2,939 cache slots to the prompt path, see below), 97 °C peak temperatures and background host processes. The numbers are not a like-for-like comparison, and this report cannot say whether its fast arm is the best result for this configuration.

## Hardware and software

- EVO-X2, AMD Ryzen AI Max+ 395 (16 cores / 32 threads), Radeon 8060S / gfx1151; 128 GB installed UMA, Linux `MemTotal` 127,098,452 KiB (121.2 GiB). GPU allocations consume the same RAM. The engine's host-to-device probe is a UMA transfer measurement, not a discrete-card PCIe-link measurement.
- Ubuntu 26.04.1 LTS, kernel `7.0.0-38-generic`, kernel amdgpu driver; ext4 on a PHISON NVMe SSD (the filesystem was checked after the run; [host-before.json](provenance/host-before.json) lists the installed drives). The recorded power mode was `performance` ([host-before.json](provenance/host-before.json)); the CPU governor and EPP were not recorded. Desktop/background host processes remained running; this was not an isolated OS or a fixed-energy comparison. The pre-existing inference server was stopped during the experiment.
- Local Release HIP build for gfx1151, CUDA/SYCL disabled, `STRATA_PREFILL_MMQ=ON`, CMake 4.2.3 and GNU C++ 15.2.0 for host code. Used AMD TheRock 7.14.1's private SDK (HIP 7.14.60850, clang 23, hipBLASLt library version 100401), and llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The running engine's [loaded libraries](provenance/loaded-sdk-libraries.json) confirm this private SDK. The host's separate HIP reports 7.16.26385 under `/opt/rocm/core-10.1`; it was not used for this build or changed by the experiment.
- Python 3.14 environment for the server and benchmark drivers; generated Python solutions used the host Python in a bubblewrap namespace, without networking/home access, with resource limits.
- Three HIP preflight tests passed: `hip_gdn_rec_head`, `hip_prefill_hcd_exact_parity`, `hip_prefill_hipblaslt_gemm`. This is not a full test-suite run. See [ctest.log](provenance/ctest.log), [build manifest](provenance/build-manifest.json) and [host snapshot](provenance/host-before.json).

## Model and configuration

- [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS), revision `38bb39ee97821de2c9009abb7e93950eec396e66`, files `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` through `00003`. Total 93,682,584,224 bytes (87.25 GiB); all three were locally SHA-256 checked against the pinned Hub file hashes ([manifest](provenance/gguf-manifest.json)). This is a mixed-format model: most expert gate/up tensors are IQ3_S and most down tensors are IQ4_NL, with selected Q8_0 tensors. The name does not mean all tensors use IQ4_XS.
- Official `tools/iq_pack.py --compat-bf16`; expert and PLE tables remain in the original GGUF shards. The conversion manifest has 460 tensors, 264 exact and 196 rounded, maximum absolute conversion error 0.014404296875. The compatibility projections use nearest-even BF16. These conversions are not lossless. The per-tensor manifests (tensor types, conversions, MTP files) are omitted from this folder; the pinned revisions and commands in [REPRODUCE.md](REPRODUCE.md) regenerate them. No new importance calibration was performed.
- MTP tensors from `Qwen/Qwen3.8-Flash-Next`, pinned revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, fetched by the official selective tool, then offline-verified, packed with Q2_0 experts and converted with `mtp_rt.py`. Used bundled `data/draft_vocab.bin` and `data/expert-profile.bin` ([protocol hashes](provenance/protocol-sha256.json); the offline verifier exits silently on success). No vision, control vectors, speed projection or custom calibration.
- Default arm: `--expert-cache auto --resident-budget-gib 55 --prefill auto --spec 4 --spec-min-p 0.5 --kv int8`, with pack, native GGUF, expert profile and MTP paths. Context 32,768 for P40 speed/correctness; 131,072 for the public long-prompt sweep and parallel tests; 262,144 for needles. Contexts of at least 65,536 add `--kv-resident 32768`. All 24,576 expert slots fit in the GPU cache (55.43 GiB); the engine reported 0.00 GiB of a separate resident CPU expert copy. File-tier expert reads still occur during prefill and are recorded in the logs.
- The engine selected 15 expert-pool workers plus its host thread on logical processors 0–15 (automatic defaults).
- Default gfx1151 exact switches remain enabled by the release. The bundled gfx1151/100401 hipBLASLt tuning table was used in all arms. The **fast arm** additionally enables the switches documented in [STRIX_HALO.md section 5](../../../docs/STRIX_HALO.md#5-what-is-not-on-by-default-it-changes-bits), plus `--lookup-chain 3 --mtp-q4 all --prefill 16384`; its complete explicit environment and arguments are in [raw/fast-long-config.json](raw/fast-long-config.json) and [raw/fast-long-env.json](raw/fast-long-env.json). These switches change rounding; the 69-task quality score below applies only to the default arm.

Exact preparation/build/run commands are in [REPRODUCE.md](REPRODUCE.md). The server configuration and explicit environment of each arm, and the command and return code of each stage, are in [raw/](raw/). The batch test's own environment (`STRATA_QFUSE=1`, `BATCH_TEST_LOG`) is set in [measure.py](scripts/measure.py). The engine startup logs of the two long-sweep arms ([default](raw/default-long-engine.log), [fast](raw/fast-long-engine.log)) show the cache size, worker placement and the prefill-borrow warning quoted here.

The engine clamps the requested CPU resident budget to currently available RAM after allocating the GPU expert cache; the resulting separate CPU expert complement is still zero because all experts fit in that cache. The fast arm also warns that its 16,384-token prefill lends 2,939 cache slots to the prompt path without a resident CPU copy of those experts, so they must be read from the files. It tests the documented fast switches/flags with this explicit memory plan, not an exhaustive search for the best plan on 128 GB UMA.

## Method

The [P40 report's script](../2026-10-05-community-2x-p40/scripts/bench.py) supplies the unchanged tasks, graders and request settings; [quality.py](scripts/quality.py) only isolates execution of generated Python. Temperature 0, top_k 1, top_p 1, seed 1, thinking off. Output caps are unchanged (reasoning 1,024, code 1,500, tools 512, completeness 512/1,024/2,048, recall 200). Two complete 69-task passes use the same server. The three P40 speed tasks run first, three repetitions each in the same cold-started process: 34 prompt/400 generated, 4,000/64 and 15,920/64. The first short request includes graph/cache warm-up and stays in the summary.

The [public V100 runner](../2026-10-04-community-v100-16gb-ram/benchmark.py) supplies exact 4,096/32,768/128,000-token synthetic Python prompts, three repetitions each, 256 generated tokens, temperature 0, reasoning off, after a separate READY warm-up. A nonce before the filler prevents prefix reuse. The default and fast arms each start a fresh engine and run serially with identical prompts and order. Neither the OS page cache nor the warmed expert cache is flushed between requests. All 27 measured speed requests reported zero reused prompt tokens, and the 10 corresponding default/fast requests (including warm-up) have identical request SHA-256 values.

The official [needle script](../../../tools/needle_bench.py) uses repository/llama.cpp text at depths 10/50/90, greedy, thinking off, 40 output tokens. Its target lengths are character estimates, so actual token counts are reported below. At the native 262,144-token context it skips `262k`, so a separate `256k` invocation covers the longest case without changing the script.

Prompt and decode rates are engine-reported (long sweep: fresh prompt tokens / engine prompt time, generated tokens / engine decode time) and exclude model loading. TTFT runs from request submission to the first nonempty generated text. Client total latency includes prefill and generation. The 27 measured speed trials, with draft counts, are in [speed.csv](analysis/speed.csv); the separate warm-up requests are in `raw/default-long/results.json` and `raw/fast-long/results.json`.

## Speed results

Each cell is median (minimum–maximum), three requests. Rates are tok/s, TTFT is seconds. See [speed.csv](analysis/speed.csv) for each trial, client total latency, draft acceptance and actual generated token counts.

| Arm / workload | Prompt tokens | Reused | Generated | Runs | Prompt tok/s | Decode tok/s | TTFT s |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| default-32k/speed-short | 34 | 0 | 400 | 3 | 88.0 (12.9–88.1) | 44.4 (36.5–44.4) | 0.43 (0.42–4.40) |
| default-32k/speed-prefill-4k | 4,000 | 0 | 64 | 3 | 676.5 (671.7–680.4) | 39.6 (39.6–39.6) | 5.96 (5.92–6.04) |
| default-32k/speed-prefill-16k | 15,920 | 0 | 64 | 3 | 729.0 (727.8–733.7) | 45.9 (45.8–45.9) | 21.88 (21.74–21.92) |
| default-long/4096 | 4,096 | 0 | 256 | 3 | 627.4 (493.0–649.1) | 42.8 (40.9–43.8) | 6.57 (6.35–8.35) |
| default-long/32768 | 32,768 | 0 | 256 | 3 | 687.2 (664.4–691.6) | 44.4 (44.0–44.7) | 47.74 (47.43–49.38) |
| default-long/128000 | 128,000 | 0 | 256 | 3 | 617.4 (617.2–617.8) | 42.1 (40.4–46.9) | 207.42 (207.26–207.47) |
| fast-long/4096 | 4,096 | 0 | 256 | 3 | 808.5 (601.4–888.2) | 40.1 (29.8–48.9) | 5.10 (4.65–6.85) |
| fast-long/32768 | 32,768 | 0 | 256 | 3 | 1,165.4 (1,161.4–1,170.4) | 49.6 (49.3–50.0) | 28.17 (28.05–28.26) |
| fast-long/128000 | 128,000 | 0 | 256 | 3 | 1,108.6 (1,106.9–1,108.8) | 47.5 (47.1–47.6) | 115.55 (115.53–115.72) |

At 128K the fast arm's median prompt throughput is 1.80× the default arm's, and median client TTFT falls from 207.42 to 115.55 seconds. This compares a bundle of settings, not one kernel change. Fast does not improve every decode cell: its 4K decode median is 40.1 versus 42.8 tok/s. The READY warm-up does not warm every verify-window graph. These prompts differ from the maintainers' own Strix Halo speed table; only the same-day arms here are matched.

## Correctness and repeatability

| Category | First pass | Second pass |
| --- | ---: | ---: |
| Reasoning | 37/40 | 37/40 |
| Python code | 12/12 | 12/12 |
| Tools / restraint | 8/8 | 8/8 |
| Completeness | 5/6 | 5/6 |
| Three-code recall | 3/3 | 3/3 |
| Total | 65/69 | 65/69 |

The two passes produced identical generated content, reasoning, tool calls and finish reasons on all 69 tasks (timings excluded). The same four failures occurred in both passes (pass/fail for every task and pass: [quality-tasks.csv](analysis/quality-tasks.csv); full responses of the first-pass failures: [failures.json](analysis/failures.json)):

| Task | Result under the unchanged grader |
| --- | --- |
| reason-33 | Computed 63 in the answer body, but reached the 1,024-token cap without the required `ANSWER:` field. |
| reason-34 | Computed 26 in the answer body, but reached the 1,024-token cap without the required `ANSWER:` field. |
| reason-39 | Reversed `calculator` as `rotalucalac`; expected `rotaluclac`. Finished normally. |
| complete-primes100 | Returned 99 primes, ending at 523; omitted the 100th, 541. Finished normally, below its 2,048-token cap. |

There were no HTTP request errors in these 138 quality requests. Tool checks cover six expected calls and two no-call cases, not a real agent's tool loop. Code checks run small generated Python functions, not repository edits. The original P40 study used GSQ-RCO IQ2_XS on CUDA/v0.1.39, while this report uses UD-IQ4_XS on HIP/v0.1.41, so it is not a controlled quantization, hardware or engine-version quality comparison. The suite has a ceiling effect and cannot rank models.

## Needle recall

| Requested length | Depth | Actual prompt tokens | Reused tokens | Result | Client seconds |
| --- | ---: | ---: | ---: | --- | ---: |
| 32k | 10% | 33286 | 0 | FOUND | 51.7 |
| 32k | 50% | 33286 | 0 | FOUND | 49.9 |
| 32k | 90% | 33287 | 16384 | FOUND | 27.2 |
| 128k | 10% | 126640 | 0 | FOUND | 205.1 |
| 128k | 50% | 126638 | 0 | FOUND | 205.5 |
| 128k | 90% | 126638 | 16384 | FOUND | 182.6 |
| 256k | 10% | 251816 | 16384 | FOUND | 441.5 |
| 256k | 50% | 251816 | 16384 | FOUND | 441.0 |
| 256k | 90% | 251817 | 16384 | FOUND | 440.9 |
| 262k | all requested depths | — | — | skipped by official script at context 262,144 | — |

9/9 completed probes found their code word, with no errors or misses; the `262k` skip is not a passed probe. The `256k` invocation restarts the script's fixed RNG, so its three code words repeat the first invocation's initial words. These probes measure recall, not general long-context reasoning.

## Parallel requests and QFUSE

The public `tools/batch_test.py` ran its two built-in prompts alone and batched (greedy, 150 output tokens, QFUSE=1, `--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow`). Its default `--mt-min 1` sets `STRATA_IQ_MT_MIN=1` for the exact reduction path; the HTTP tests below do not set it. Token streams matched in 2/2 slots (150 tokens each); the token IDs are in `raw/batch-exact.json`.

Four fresh HTTP servers tested QFUSE on/off × prefill-borrow on/off with two simultaneous distinct ~19,600-token prompts and up to 256 output tokens. The prompts are the public synthetic Python filler, not #1356's unpublished original prompts. `--batch-mtp` was not tested.

| QFUSE | Prefill borrow | Requests with nonempty text / submitted | Finish reasons | Client seconds (slot 0 / 1) |
| --- | --- | ---: | --- | --- |
| 1 | on | 2/2 | length / length | 62.2 / 73.9 |
| 1 | off | 2/2 | length / length | 89.3 / 68.6 |
| 0 | on | 2/2 | length / length | 62.1 / 73.9 |
| 0 | off | 2/2 | length / length | 69.6 / 90.2 |

All eight requests returned 256 tokens with no transport errors, hangs or obvious garbling on inspection. Within each prefill-borrow setting, toggling QFUSE gave identical generated text in both slots ([comparison](analysis/parallel-comparison.json)). Toggling prefill borrow changed the wording; no semantic oracle scored the outputs. This is one paired trial per cell. A `length` finish is expected at the 256-token cap.

## Memory, temperature and limits

Two-second host samples recorded a minimum `MemAvailable` of 39.14 GiB, peak GTT of 67.92 GiB and peak VRAM carve-out use of 0.70 GiB. These are whole-host counters, not process RSS, and GTT is already part of the UMA RAM. Swap was in use throughout: used swap at the start and end of each stage was 6.58–7.21 GiB (32 GiB total, from the `SwapFree` values in `analysis/summary.json`; the in-stage samples are not included). Swap I/O was not measured, so this is not a no-paging claim. No OOM or unexpected engine exit occurred. Five-second hwmon samples peaked at 97.0 °C (amdgpu) and 98.1 °C (k10temp); the amdgpu-reported power maximum was 137.0 W, which on this APU is not a discrete-GPU or whole-system energy figure. See the [thermal summary](analysis/thermal-summary.json); the raw samples are not included.

Only this model and build, one machine on one day, and the listed inputs were measured. Not run: a long multi-turn soak, a real agent harness, vision, a fast-arm correctness suite, KL/perplexity calibration, CPU-only parity, Vulkan/CUDA/SYCL comparison, and a same-day comparison with another local runtime. This report does not establish the fast arm's quality equivalence, a fix for long-session engine stops such as #1500 ("Engine stopped unexpectedly on strix halo"), or a recommendation to replace another runtime.

## Evidence

- [Summary JSON](analysis/summary.json), [individual speed requests](analysis/speed.csv), [failed-task responses](analysis/failures.json), [server configurations, environments, stage log and long-sweep engine logs](raw/), [reproduction](REPRODUCE.md).
- Not included, to keep the folder small: per-request responses and grader output, needle and parallel request/SSE records, the other engine and per-stage logs, memory/thermal telemetry samples, and the per-tensor manifests. See [TRIMMED.md](TRIMMED.md).
- [Build](provenance/build-manifest.json), [model hashes](provenance/gguf-manifest.json), [source/script/profile hashes](provenance/protocol-sha256.json), [needle corpus](provenance/needle-corpus-manifest.json), [HIP test log](provenance/ctest.log).
- [SHA256SUMS](SHA256SUMS) covers the report, scripts and retained evidence. Engine source is unchanged by this report.
