# Strata standard benchmark: NVIDIA GeForce RTX 3060

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T082538Z-6302f9ef`, started 2026-10-20T08:25:38Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA GeForce RTX 3060 (12.0 GiB dedicated)
- CPU: Intel(R) Core(TM) Ultra 7 265; RAM: 46.2 GiB
- Storage of the model files: nvme Lexar SSD NM1090 PRO 2TB (nvme)
- OS: Ubuntu 24.04.5 LTS 7.0.0-38-generic; drivers: gpu-driver 615.71.09; runtime: cuda 13.4
- Strata commit: `a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e`; engine 0.1.46; release build
- Background work: none besides sshd

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (release 0.2.1 terminal+engine 0.1.46 cuda12, hand-written run config mirroring setup defaults (IQ3_XXS, int8 KV, MTP spec 4); box: p2003)
- Context 32768 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 3 min 22 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 183.0 (181.5-188.4) n=3 | 59.1 (58.2-63.0) n=3 | 0.45 (0.43-0.46) n=3 | 3.97 (1.74-4.43) n=3 | 8.7 (5.6-9.1) n=3 |
| prompt-4k | 961.5 (958.7-962.0) n=3 | 54.6 (51.1-62.3) n=3 | 4.30 (4.29-4.32) n=3 | 7.69 (7.69-7.69) n=1 | 13.6 (12.5-14.3) n=3 |
| prompt-16k | 1100.0 (1098.3-1103.7) n=3 | 53.2 (52.3-54.1) n=3 | 14.94 (14.88-14.95) n=3 | 23.29 (23.29-23.29) n=1 | 24.5 (24.4-24.7) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 3 | 6 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 3 | 9 | 0 | 0 |

Sanity checks: **12 of 12**.

| Case | Result | Reason |
| --- | --- | --- |
| I1 | pass | the expected JSON value |
| I2 | pass | the expected JSON value |
| I3 | pass | the expected JSON value |
| I4 | pass | the exact expected text |
| R1 | pass | the exact expected text |
| R2 | pass | the exact expected text |
| R3 | pass | the exact expected text |
| R4 | pass | the expected JSON value |
| T1 | pass | the expected tool call |
| T2 | pass | the expected tool call |
| T3 | pass | the expected tool call |
| T4 | pass | the expected tool call |

Concurrency rounds (four requests released together):

| Round | Wall s | Output tokens | Aggregate tok/s | Failed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 38.693398 | 2048 | 52.928926 | 0 |
| 2 | 32.637462 | 1840 | 56.376933 | 0 |
| 3 | 38.39188 | 1994 | 51.938066 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r1-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r3: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
