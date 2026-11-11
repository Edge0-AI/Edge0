# Strata standard benchmark: NVIDIA RTX A4000, NVIDIA RTX A4000

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T091531Z-93d2b51e`, started 2026-10-20T09:15:31Z. **Status: complete.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

## Hardware and software

- GPU: NVIDIA RTX A4000 (16.0 GiB dedicated); NVIDIA RTX A4000 (16.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2643 v3 @ 3.40GHz; RAM: 62.7 GiB
- Storage of the model files: nvme Lexar SSD NM620 512GB (nvme)
- OS: Ubuntu 24.04.4 LTS 6.8.0-147-generic; drivers: gpu-driver 580.178.04; runtime: cuda 12.9, cuda-driver-supports 13.0
- Strata commit: `a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e`; engine 0.1.46; release build
- Background work: none besides sshd

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (release 0.2.1 terminal+engine 0.1.46 cuda12, hand-written run config mirroring setup defaults (IQ3_XXS, int8 KV, MTP spec 4); box: BOX)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 24 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 135.4 (129.8-144.1) n=3 | 90.3 (81.0-97.1) n=3 | 0.59 (0.55-0.61) n=3 | 2.42 (1.20-3.10) n=3 | 5.8 (5.1-6.2) n=3 |
| prompt-4k | 1158.0 (1149.1-1172.7) n=3 | 74.0 (72.5-74.0) n=3 | 3.56 (3.51-3.58) n=3 | 10.16 (10.16-10.16) n=1 | 10.5 (10.5-10.6) n=3 |
| prompt-16k | 1852.1 (1828.5-1901.3) n=3 | 75.8 (70.8-80.9) n=3 | 8.88 (8.65-8.99) n=3 | 14.12 (13.13-15.11) n=2 | 15.4 (15.3-16.1) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 1 | 8 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 12 | 2 | 10 | 0 | 0 |

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
| 1 | 28.329844 | 2048 | 72.291256 | 0 |
| 2 | 27.663746 | 1930 | 69.766401 | 0 |
| 3 | 25.367314 | 1913 | 75.412005 | 0 |

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: concurrency-r1-c2: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c1: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r2-c4: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c3: first_answer_s is not available for this request
- Excluded from speed summaries: concurrency-r3-c4: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r2: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The server had answered no request before this run (a freshly loaded engine).
