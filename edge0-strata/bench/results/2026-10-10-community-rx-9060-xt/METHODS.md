# Detailed methods for the RX 9060 XT benchmark

Measured October 9–10, 2026. Contributor: [isoux35](https://github.com/isoux35).

On this Windows desktop, increasing IQ2_XS prefill from 2,048 to 8,192 reduced median fresh-prompt latency by about 56% for the two tested prompt sizes. In a separate same-day coding comparison at matched 32K settings, IQ2_XS completed the small synthetic tasks sooner than IQ3_XXS, with both passing the acceptance checks. A single near-limit recall request also completed with 113,501 input tokens under a 122,880-token context limit.

These are measurements of this installed stack, not a universal ranking of the models or a claim of reliable coding over 120K tokens. No engine code, defaults, or upstream headline performance claims are changed.

## Hardware and software

| Component | Tested setup |
| --- | --- |
| GPU | AMD Radeon RX 9060 XT, 16 GB VRAM, gfx1200; one discrete GPU used |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| RAM | 64 GB installed, approximately 61.66 GiB usable by Windows; DDR5-6000 |
| Operating system | Windows 11 Home 25H2, build 26200.9457 |
| Display driver | 32.0.31041.1004 |
| Engine | Prebuilt Strata 0.1.40.3, windows-x64, HIP backend |
| Packaged runtime | BUILD.json records ROCm `10.2.0a20260930`, hipBLASLt version `100500` |
| Wrapper checkout | `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` |
| Client stack | OpenCode 1.18.34 in Ubuntu 24.04 WSL; OpenChamber 1.24.2 frontend |
| Independent test runner | Node.js v24.21.0 in WSL |

The engine's exact compiled source commit is not recorded; the wrapper checkout is **not** presented as the binary's build commit. Storage model, negotiated PCIe link, GPU power limits, temperature, and detailed CPU-worker defaults were not recorded. This was not an isolated benchmark workstation: browser, client services, WSL, and normal background applications were present.

This uses native Windows HIP, **not** Vulkan inference or an engine running in WSL. Strata distributes experts and context storage across GPU and system memory; this is not an all-weights-in-VRAM configuration.

## Model artifacts

Both models came from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).

| Quant | GGUF filename | Bytes |
| --- | --- | ---: |
| IQ2_XS | Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf | 39,225,954,592 |
| IQ2_XS | Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf | 28,800,138,432 |
| IQ3_XXS | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf | 47,039,860,096 |
| IQ3_XXS | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf | 28,800,138,432 |

The IQ2_XS first-shard download metadata records revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` and object ETag `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7`. The shared second shard was reused from the existing installation; its original revision and the IQ3_XXS revision were not recorded. File sizes alone do not establish artifact identity.

The existing prepared IQ packs and expert profile were used. MTP used the prepared runtime pack from `mtp-q2_0.gguf` (889,014,272 bytes). Pack, draft, and expert-profile hashes were not recorded during the trials, limiting exact cross-machine reproducibility. No model binaries, packs, or calibration files are included here. No new calibration was forced for these tests.

## Common settings and timing boundaries

The tested arguments below use portable path labels instead of the contributor's file paths. These are engine argument details, not a complete standalone server launch command:

```text
STRATA_HIP_WMMA=1
--pack <PREPARED_PACK>
--native <FIRST_GGUF_SHARD>
--ple-gguf <SECOND_GGUF_SHARD>
--expert-profile <EXPERT_PROFILE>
--expert-cache auto
--prefill <ARM_SIZE>
--spec 4
--spec-min-p 0.5
--mtp <PREPARED_MTP_RUNTIME_PACK>
--max-context <CONTEXT_LIMIT>
--kv int8
--vram-reserve-mib 2048
--kv-resident 32768
```

One parallel request; vision and experimental speed projection disabled. Sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0, seed 42, repetition penalty 1.0, presence and frequency penalties 0.

Direct recall trials used low reasoning and a 1,024-token output cap, streaming requests, `cache_prompt=false`, and `strata_checkpoint=false`. All measured recall requests reported zero reused prompt tokens. Streaming first-token latency starts at client request submission and ends at the first nonempty reasoning, answer, or tool-call delta; first-answer latency excludes reasoning. End-to-end wall time includes HTTP and output reception but excludes model loading. Engine prompt and decode timing are reported separately, and decode counts include reasoning tokens.

Available physical RAM was sampled on Windows every 0.5 seconds for direct requests. Coding resource summaries use whole-host/whole-adapter counters, not process-isolated memory. Dedicated and shared GPU counter peaks are not simultaneous allocations and must not be added together or treated as extra VRAM capacity.

## Prefill comparison with IQ2_XS

Context limit was 122,880 for all four arms. Each arm started a fresh engine, followed by one excluded quote-arithmetic warmup, then three repetitions of the 8K prompt and three of the 28K prompt in that order. No OS file-cache purge or adaptive expert-cache reset was performed between requests. These are **fresh prompts without prefix reuse**, not fully cold model/expert-cache measurements. Arm order was 2,048, 4,096, 8,192, then 16,384; it was not randomized.

Actual input lengths were 8,019 and 28,180 tokens. Completion lengths ranged from 111 to 120 tokens. Entries show median wall seconds with min–max in parentheses; throughput columns are median engine tokens/second. Each row has three repetitions per prompt size.

| Prefill | 8K wall seconds | 8K prompt / decode tok/s | 28K wall seconds | 28K prompt / decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 25.400 (25.281–25.765) | 346.3 / 53.2 | 83.355 (82.183–84.013) | 347.0 / 57.9 |
| 4,096 | 15.703 (15.682–15.772) | 593.3 / 52.7 | 49.558 (49.393–50.104) | 593.1 / 57.0 |
| 8,192 | 11.212 (11.111–11.423) | 879.4 / 56.3 | 37.008 (36.915–37.021) | 807.4 / 57.6 |
| 16,384 | 11.384 (11.189–11.428) | 876.3 / 55.3 | 36.778 (36.757–37.087) | 811.2 / 57.8 |

All 24 measured recall answers contained the correct three values, but **0/24 were strict plain JSON** because they used Markdown fences. Minimum available RAM per arm was 9.17, 9.03, 8.76, and 9.02 GiB respectively.

8,192 was selected locally: approximately 56% lower median wall time than 2,048 on each tested size, with essentially unchanged decode speed. 16,384 did not meaningfully improve latency. The selection rule required at least 5% improvement over the smaller selected arm and more than 4 GiB available RAM. This does not justify changing defaults for other hardware or workloads.

Per-run throughput ranges, timing, token counts, answers, and warmups are in [data](data); `scripts/summarize.py` recomputes medians and ranges without contacting a model.

## Matched coding comparison at 32K

Both quants were tested the same day with context 32,768, prefill 8,192, INT8 KV, resident KV 32,768, WMMA enabled, 2,048 MiB VRAM reserve, spec 4, one request at a time, and the same sampling settings. Coding used medium reasoning and an 8,192-token output allowance. IQ2 was tested first, then IQ3; order was not randomized. Fresh sessions and copied fixtures were used for each trial, while adaptive expert cache could remain warm within a model's batch.

Three synthetic TypeScript tasks were repeated three times each per model. The fixtures require an interval-merge repair, an order-summary refactor, and a three-file reservation-workflow repair. The Build agent was limited to 12 steps and at most two repair rounds, with only the named source files editable and only the acceptance-test command allowed. Tests, instructions, and client config were checked for changes, and tests were also run independently after completion. The test totals below count repetitions: **81 checks across the three fixtures, repeated three times = 243**, not 243 distinct tests.

| Measurement | IQ2_XS | IQ3_XXS |
| --- | ---: | ---: |
| Tasks passed | 9/9 | 9/9 |
| Acceptance checks across repetitions | 243/243 | 243/243 |
| Overall median task seconds | 99.988 | 136.321 |
| Overall task range, seconds | 55.463–145.326 | 81.012–166.632 |
| Weighted engine decode tok/s | 49.61 | 42.52 |
| Minimum host available RAM, GiB | 8.92 | 3.11 |
| Peak adapter dedicated counter, GiB | 14.88 | 15.04 |
| Peak adapter shared counter, GiB | 34.16 | 41.12 |

| Task | IQ2 median seconds (min–max) | IQ3 median seconds (min–max) |
| --- | ---: | ---: |
| Merge windows | 60.368 (55.463–99.988) | 109.951 (81.012–136.321) |
| Refactor orders | 93.343 (91.803–120.835) | 119.482 (104.944–137.212) |
| Reservation workflow | 143.302 (125.702–145.326) | 158.644 (151.560–166.632) |

IQ2's overall median task time was about 27% lower and weighted decode speed about 17% higher, with approximately 5.8 GiB more minimum RAM headroom. Both solved these limited tests; this does not establish equal general reasoning or production-code quality. Engine totals can include client helper requests and reasoning, so weighted decode is `sum(output tokens) / sum(decode seconds)`, not a pure single-request model benchmark. This is **not** a comparison against the older dense Qwen 27B llama.cpp setup.

All 18 trial records are in [coding-per-run.json](data/coding-per-run.json). Synthetic original [fixtures](fixtures) and all 18 generated [solutions](solutions), with per-file SHA-256 hashes in the records, permit independent checks. Private client config and full agent transcripts are intentionally excluded.

## Large context and cache reuse

The selected IQ2 profile used context **122,880**, resident KV 32,768, and prefill 8,192. One recall request contained **113,501 input tokens** and generated 119 tokens, with no prefix reuse. Three fixed markers near the start, middle, and end were recovered correctly.

| Single near-limit recall | Measurement |
| --- | ---: |
| Wall time | 146.629 s |
| First streamed token, including reasoning | 144.157 s |
| First answer content | 145.537 s |
| Engine prompt processing | 789.9 tok/s; 143.696 s |
| Engine generation | 46.9 tok/s; 2.539 s |
| MTP draft offered / accepted | 98 / 85 |
| Minimum available RAM | 9.49 GiB |

The answer was semantically correct but Markdown-fenced, hence failed strict JSON formatting. Input plus an 8,192-token reserve fits the configured context arithmetically; this request actually used a 1,024-token cap and generated only 119. It did **not** test an 8K answer at the limit, complex long-context reasoning, or multi-file coding with 120K history. This was one run, not a distribution.

An OpenCode backend cache check used a different approximately 22K-input document followed by three short questions in the same session. The first assistant response took 62.468 seconds with 22,102 uncached tokens; follow-ups took 1.276, 1.181, and 1.085 seconds while reusing 22,205, 22,269, and 22,323 tokens. All four answers passed. These latencies come from assistant creation/completion timestamps, excluding the five-second polling interval. This was an API/backend test, not a timed OpenChamber UI benchmark or an uncached-versus-cached workload with identical prompts.

Three final medium-reasoning smoke cases each ran once with an 8,192-token allowance: quote arithmetic (6.870 s), validation classification (5.343 s), and the correct `read_file` tool call (4.367 s). The first two used `response_format: {"type":"json_object"}` and passed semantic and strict JSON checks. Strata's response handling can buffer, validate, and canonicalize JSON, so their observed first-answer time is not raw engine first-token latency. This is not grammar-constrained decoding. The tool call was checked but **not executed**.

## Incidents and stability limits

Successful-run measurements do not include these recovered setup/client incidents:

- One undelivered coding submission returned HTTP 502 through the local client proxy. The session was verified empty and idle twice before that undelivered prompt was sent once. No completed task was silently repeated. The preceding helper activity can affect expert-cache warmth and aggregate counters.
- One read-only client check returned HTTP 502 after an instance reload. Only read-only GET checks were retried; task submissions and configuration writes were not automatically replayed.
- An initial prefill cleanup idle race occurred after a completed answer. The harness added an idle guard and resumed only unsubmitted trials.
- A Windows CRLF versus canonical LF fixture-hash mismatch was corrected before prompts were sent, without changing acceptance tests.
- Fixture policy blocked attempts to run disallowed shell commands. These expected permission errors were not model crashes; counts remain in the trial records.

No relevant Windows crash, display-driver, or WHEA events were observed during the approximately 72-minute tuning window. This is not an overnight burn-in, proof of hardware stability, or a guarantee of future crash-free use. These observations are not attributed to an engine bug or to quantization without further evidence.

## Reproduction

Start with Strata's supported setup and prepare the matching model and MTP packs. Use the tested 0.1.40.3 package for a version-matched comparison; a newer release is a separate experiment. Apply the argument settings above with paths and the HIP device ordinal appropriate to your computer. Do not copy private configuration from another machine or expose an unauthenticated server on the network.

The exact direct prompts and scoring are in `scripts/reproduce_direct.py` and `scripts/benchmark.py`. The wrapper is adapted for portable command-line paths and an optional `STRATA_API_KEY` environment variable; it does not start/restart servers, modify configs, install software, or execute generated tools. Start a fresh engine separately for each prefill arm, change only its prefill argument, and verify its setting before running:

```text
python scripts/reproduce_direct.py --base http://127.0.0.1:8081 --model qwen3.8-flash-next-iq2_xs --mode prefill --prefill 8192 --output new-prefill-8192.json
python scripts/reproduce_direct.py --base http://127.0.0.1:8081 --model qwen3.8-flash-next-iq2_xs --mode recall120k --output new-recall120k.json
python scripts/reproduce_direct.py --base http://127.0.0.1:8081 --model qwen3.8-flash-next-iq2_xs --mode medium --output new-medium.json
python scripts/summarize.py
```

The direct harness runs on Windows because it samples Windows physical memory. Output files must not already exist; failed requests are preserved and not automatically retried. The `--prefill` value labels the chosen arm; it does **not** configure the server. Near-limit tests require adequate available RAM and can keep the computer busy for minutes. Run locally with other model work idle.

For the coding comparison, copy one original fixture to a fresh disposable project/session for each model, task, and repetition. Use the same matched 32K settings, the Build agent limits above, and this prompt, substituting the fixture name and repeat number:

```text
Independent coding trial {case}, repeat {repeat}. Read AGENTS.md, README.md, all existing source files and acceptance.test.ts. Complete the task in README, preserving the public API and the tests. Use tools to edit only permitted source files and run node --test acceptance.test.ts. If tests fail, make at most two repair rounds. Report the actual test counts and a concise explanation. No network, packages, delegation or other projects.
```

Use identical client provider options for both models: medium reasoning, 8,192 output, interleaved `reasoning_content`, temperature 1.0/top-p 0.95, and the same model for helper requests. Do not reuse real project sessions. Run `node --test acceptance.test.ts` independently, check the protected files, and for the refactor also confirm that the original exported function imports and calls the two helpers. Original fixtures intentionally contain bugs and may fail until implemented.

To check a published solution, copy its source files over a **disposable copy** of the matching original fixture, then run the acceptance test. Do not overwrite your own project. Record client/engine versions, actual token counts, all repetitions, failures, and timing boundaries for any new comparison.

## Included files and privacy

Only benchmark prose, whitelisted numeric/results data, synthetic prompts/tests, generated synthetic solutions, and small reproduction scripts are included. Paths, credentials, network addresses belonging to the contributor, session/message IDs, real project files, raw Windows logs, screenshots, and conversation histories are excluded. Loopback URLs and portable path labels in the reproduction instructions are generic examples.
