# Strata standard benchmark: NVIDIA RTX A4000, NVIDIA RTX A4000

Protocol `strata-qwen38-thinking-v1-draft.1` (draft), run `20261020T081800Z-187f1d5a`, started 2026-10-20T08:18:00Z. **Status: nonconforming.**

Written by `strata bench run`. The machine-readable result is [results.json](results.json); every request, its raw usage and timings and the client event offsets are in [outputs.jsonl](outputs.jsonl).

This run is not a complete standard result:

- the Strata commit is unknown (not a git checkout and no COMMIT file)

## Hardware and software

- GPU: NVIDIA RTX A4000 (16.0 GiB dedicated); NVIDIA RTX A4000 (16.0 GiB dedicated)
- CPU: Intel(R) Xeon(R) CPU E5-2643 v3 @ 3.40GHz; RAM: 62.7 GiB
- Storage of the model files: not measured
- OS: Ubuntu 24.04.4 LTS 6.8.0-147-generic; drivers: gpu-driver 580.178.04; runtime: cuda 13.0
- Strata commit: `not measured`; engine 0.1.46; release build
- Background work: none besides sshd

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs; quantization IQ3_XXS; revision not measured; source revision not measured
- Files: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` `219ea929900dfa9e...`; `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` `316b46f3a2dbd68c...`; `expert-profile.bin` `8f59b4aa8873209d...`
- Track: **custom** (release 0.2.1 terminal+engine 0.1.46 cuda12, hand-written run config mirroring setup defaults (IQ3_XXS, int8 KV, MTP spec 4); box: z440)
- Context 65536 tokens; server capacity 1 slot(s)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0, reasoning_effort xhigh; thinking on, preserved
- Engine options: `config.json`; chat template: `provenance/chat_template.jinja`

## Method

The frozen requests of `bench/prompts/standard-v1` (corpus `e75eff8b9feea04f...`) were sent over loopback HTTP, streaming, one at a time, in this order: 1 warm-up (excluded), 3 repetitions of the short, 4K and 16K prompts, and the twelve sanity checks. Prompt and decode speed are the engine's own `timings`; time to first token is the client's monotonic clock from dispatch to the first nonempty reasoning, answer or tool-call delta. No request was retried. Time after the model loaded: 2 min 18 s.

## Results

| Case | Prompt tok/s | Decode tok/s | TTFT s | First answer s | Total s |
| --- | --- | --- | --- | --- | --- |
| short | 131.0 (130.9-144.1) n=3 | 91.1 (87.6-97.3) n=3 | 0.61 (0.55-0.61) n=3 | 1.71 (1.21-1.74) n=3 | 4.7 (4.2-5.8) n=3 |
| prompt-4k | 1157.9 (1148.2-1173.5) n=3 | 75.5 (71.9-87.6) n=3 | 3.56 (3.51-3.59) n=3 | 7.60 (6.26-8.95) n=2 | 10.3 (9.4-10.6) n=3 |
| prompt-16k | 1856.9 (1831.0-1900.5) n=3 | 75.2 (72.3-83.8) n=3 | 8.86 (8.66-8.98) n=3 | 12.05 (12.05-12.05) n=1 | 15.8 (14.7-15.9) n=3 |

Cells are median (min-max) n=samples.

| Phase | Planned | Completed | Truncated | Failed | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup | 1 | 0 | 1 | 0 | 0 |
| performance | 9 | 2 | 7 | 0 | 0 |
| sanity | 12 | 12 | 0 | 0 | 0 |
| concurrency | 0 | 0 | 0 | 0 | 0 |

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

## Failures, exclusions and limitations

- No request failed.
- Excluded from speed summaries: performance-prompt-16k-r2: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-16k-r3: first_answer_s is not available for this request
- Excluded from speed summaries: performance-prompt-4k-r1: first_answer_s is not available for this request
- Prompt-prefix reuse is the shared template prefix only; the cache was not emptied between requests, and the expert cache stays warm and evolves. These are not cold-model measurements.
- Output caps include reasoning; a request that reaches its cap is `truncated` and fails its sanity check.
- The runner cannot see other programs on the machine, the repeat-stop and loop-recovery switches of the server, or the quantization repository's revision; see `missing` in results.json for every value it could not read.
- The runner could not tell whether the server was freshly loaded.
