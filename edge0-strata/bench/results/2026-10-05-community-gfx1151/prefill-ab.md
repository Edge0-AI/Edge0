# Controlled Radeon 8060S (`gfx1151`) HIP comparison

Measured on 2026-10-05 on a Radeon 8060S / Ryzen AI Max+ 395 system with 128 GB of unified memory. The model was
Qwen3.8-Flash-Next GSQ-RCO Q2_0. All 24,576 routed experts were resident in the 31.64 GiB GPU expert cache. KV
streaming kept 32,768 of 131,072 cells per QSA layer in VRAM and used 1.55 GiB of pinned RAM.

The source revision was `a68d0b7fd88c206d55c2ca1d3c2811300b07bd14`. It contains the `gfx1151` enablement from
PR #895, the opt-in dense MMQ path from PR #820, and a `gfx1151` hipBLASLt 1.5.0 tuning table adapted from the work
in PR #755. PR #855 was not included. The HIP runtime reported version `7.17.26392-0000000`; the tuning table is for
hipBLASLt version `100500` and contains 32 rows.

## Method

`tools/hip/bench_prefill.py` sent a warm-up followed by fresh prompts containing 140, 280, 140 and 280 generated
JavaScript functions. Every fresh prompt had zero reused KV tokens. Each fresh request was followed by one cached
request. Sampling was fixed at temperature 0, top-k 1, top-p 1, min-p 0 and seed 42, with reasoning disabled. Every
measured request generated 128 tokens and stopped at the intentional length limit.

The server was restarted between arms. `STRATA_SH_STREAM` defaults to enabled in this revision, so C and D are the
same effective configuration; D merely sets the default explicitly.

| Arm | Runtime environment | Result file | Engine log |
| --- | --- | --- | --- |
| A | `STRATA_DENSE_MMQ=0`, no tuning table; SH stream default on | [`gfx1151-plain.json`](gfx1151-plain.json) | [`engine-A.log`](engine-A.log) |
| B | A plus the `gfx1151` hipBLASLt tuning table | [`gfx1151-B-tuningTables.json`](gfx1151-B-tuningTables.json) | [`engine-B.log`](engine-B.log) |
| C | B plus `STRATA_DENSE_MMQ=1`; SH stream default on | [`gfx1151-C-tuningTables-denseMMQ.json`](gfx1151-C-tuningTables-denseMMQ.json) | [`engine-C.log`](engine-C.log) |
| D | C plus explicit `STRATA_SH_STREAM=1` | [`gfx1151-D-tuningTables-denseMMQ-PR826.json`](gfx1151-D-tuningTables-denseMMQ-PR826.json) | [`engine-D.log`](engine-D.log) |
| E | C with `STRATA_SH_STREAM=0` | [`gfx1151-E-tuningTables-denseMMQ-noPR826.json`](gfx1151-E-tuningTables-denseMMQ-noPR826.json) | [`engine-E.log`](engine-E.log) |

The engine logs confirmed that B-E loaded 32 tuning rows for `gfx1151` / `100500`, and that C-E enabled dense MMQ.

## Fresh-prompt results

The table reports the engine's prompt-processing rate. "First" and "warmed" describe execution order within an
arm; zero KV reuse does not imply a cold filesystem or GPU-library cache.

| Fresh prompt | A: plain hipBLAS | B: tuned hipBLASLt | C: + dense MMQ | D: SH stream on | E: SH stream off |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,210 tokens, first | 261.6 | 549.7 | 554.8 | 567.4 | 568.9 |
| 8,830 tokens, first | 250.9 | 525.0 | 526.0 | 527.0 | 528.1 |
| 4,210 tokens, warmed | 247.5 | 526.1 | 528.5 | 527.6 | 526.0 |
| 8,830 tokens, warmed | 247.7 | 524.6 | 523.4 | 524.7 | 524.2 |
| **Weighted rate** | **250.91** | **528.91** | **529.96** | **532.43** | **532.58** |

Rates are tokens per second. The weighted row divides the 26,080 fresh tokens in each arm by their combined engine
prefill time.

The tuning table raised weighted fresh-prompt throughput from 250.91 to 528.91 tok/s: **2.11x, or 110.8% faster**.
The four matched improvements were 109-113%, so the effect was present at both prompt sizes and on both passes. The
combined wall time of the four fresh requests, including their 128-token answers, fell from 114.50 to 59.97 seconds
(47.6%).

Dense MMQ changed the weighted B result by only +0.2%. C-E differ by less than 0.5%, which is consistent with normal
run-to-run variation rather than a measurable prefill benefit from dense MMQ or SH streaming on this workload.

## Shared-expert stream result

D and E isolate `STRATA_SH_STREAM`: their prompts, generated responses and MTP draft behavior matched. Across all
eight measured requests, excluding the warm-up:

| Configuration | Generated tokens | Decode time | Weighted decode rate |
| --- | ---: | ---: | ---: |
| D: SH stream on | 1,024 | 21.894 s | 46.77 tok/s |
| E: SH stream off | 1,024 | 22.441 s | 45.63 tok/s |

Enabling the shared-expert stream improved weighted decode throughput by **2.5%** and saved 0.547 seconds of decode
time. All eight individual requests were faster with it, by about 1.3-3.5%. As expected for a decode optimization,
it did not materially change fresh-prompt throughput.

## Limits

This is one ordered benchmark session on one machine, not a confidence interval or a claim for other GPUs and
models. C and D provide one repeat of the same effective configuration and show the approximate timing noise. The
cached follow-ups are retained in the JSON files, but their fresh-token counts differ between some earlier arms
because the generated replies differ; they are therefore not used for the hipBLASLt prefill comparison.
