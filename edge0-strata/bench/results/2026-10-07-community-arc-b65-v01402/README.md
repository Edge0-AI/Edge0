# Arc Pro B65: Strata v0.1.40.2 on PCIe Gen4

Measured on 2026-10-07 by timnevits. **Official upstream release**, commit
`e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`, with **no local engine patches**.
The measurements come from my B65 hardware.

Both memory profiles pass their bounded correctness checks. With the same 7K inputs, the 8K/prefill-512 profile delivers **43.49 decode tok/s**, versus **42.38** with the 262K/prefill-512 profile (-2.6%). The 8K/prefill-4096 profile reduces median 7K TTFT from **22.23 to 8.47 seconds**, with decode tradeoffs below.

## Hardware and configuration

One Intel Arc Pro B65, 32 GiB VRAM (`8086:e222`), **Gen4 x16**, 200 W cap;
i5-12600K, 128 GB DDR4-3200, Samsung 980 NVMe. Other model owners were stopped.
Ubuntu 26.04.1, kernel 7.0.0-38, xe/NEO 26.22.38646.7, Level Zero 1.28.6.
Native Release/JIT build with existing oneAPI 2026.1.0 and oneMKL, precise FP,
correctly rounded divide/sqrt and 32-lane subgroups. The tagged engine labels
itself `0.1.40-sycl`; the commit and binary hash identify this build.
Exact metadata: [system.json](system.json), [build.json](build.json).

Original **full 512-expert Flash-Next IQ2_XS**, not Coder:
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Original Q2_0 MTP pack from
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
[artifacts.json](artifacts.json) identifies both shards, retained dense pack,
tokenizer/template, draft and expert ranking. Ranking matches the release
byte for byte. The model has a native 262144-token window. The tables compare **8192 and
262152 native context allocations (8K/262K profiles)**, using matched short and 7K inputs. The API reserves eight verification slots,
leaving 8184 and 262144 usable combined tokens respectively. No model, quantization, driver, compiler or ranking changes
between configurations.

One request, INT8 resident KV, four workers, automatic hot
expert cache and streaming experts. Every missing expert is in a protected
pinned host mirror (16 GiB cap). `STRATA_VERIFY_NO_HOST=1` keeps computation
on the GPU, including direct reads from that mirror. Native verify/commit
graphs, MTP window 4/max 4/min-p 0.5 (up to three drafts). Thinking off,
greedy/seed 17; prompt, conversation and suffix caches, short-read and expert
adaptation off. `STRATA_DBG_NAN` and QFUSE absent. Allocation alias checking
keeps its upstream default. The two 8K profiles reserve 3072 MiB and differ only in prefill batch size:
[512](profile-8k-512.json), [4096](profile-8k-4096.json). The
[262K profile](profile-262k-512.json) uses prefill 512, reserve 4096 MiB and
explicit `--no-prefill-borrow`, preserving the smaller window's implicit
no-borrow policy. It reserves more KV and has fewer hot expert slots; this is
a comparison of the complete memory profiles, not context as a single variable.
[environment.json](environment.json) is shared.

Actual startup allocations below are identical across each profile's three main
runs. Expert slots count per-layer experts, rather than the model's 512 experts
per MoE layer. They help explain the memory/performance tradeoff.

| Profile | Hot expert slots | Hot cache MiB | Pinned missing-expert mirror GiB | Free VRAM at startup MiB |
| --- | ---: | ---: | ---: | ---: |
| 8K / 512 | 17881 | 24588 | 9.01 | 3197 |
| 8K / 4096 | 16160 | 22207 | 11.33 | 4237 |
| 262K / 512 | 14388 | 19761 | 13.72 | 4185 |

## Three-run results

Main suite: **three fresh processes per profile**, five fixed tasks at each
of 512 and 7000 input tokens, 640 generated tokens. An unmeasured full-shape
warmup precedes each input tier. Tasks cover code, prose, inventory and incident
planning; padding precedes the instruction. Each profile has 48 total requests,
30 measured. For each table row, take the median of five tasks within each run,
then the median and range of those three run medians. Individual cells are
available in the raw data.

Public-fixture suite: three fresh processes per profile, 2185-token request
first, then 20-token request, 256 outputs each, followed by an isolation canary.
No separate warmup; any first-request cost stays in its measurement. This is
9 total requests/6 measured per profile. Its prompts and output lengths differ
from the main suite.

| Native context allocation | Prefill | Input/workload | Output | Runs | Prompt tok/s | Decode tok/s | TTFT seconds |
| ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| 8192 | 512 | 512, five tasks | 640 | 3 | 352.37 (352.28–352.81) | 45.01 (44.99–45.02) | 1.486 (1.485–1.487) |
| 8192 | 512 | 7000, five tasks | 640 | 3 | 315.44 (315.31–315.46) | 43.49 (43.49–43.49) | 22.226 (22.225–22.236) |
| 8192 | 512 | 20, public fixture | 256 | 3 | 54.60 (54.56–54.61) | 54.48 (54.47–54.49) | 0.400 (0.400–0.401) |
| 8192 | 512 | 2185, public fixture | 256 | 3 | 323.36 (323.30–323.43) | 49.72 (49.72–49.72) | 6.817 (6.816–6.818) |
| 8192 | 4096 | 512, five tasks | 640 | 3 | 330.92 (330.88–331.11) | 44.64 (44.61–44.65) | 1.581 (1.580–1.581) |
| 8192 | 4096 | 7000, five tasks | 640 | 3 | 830.00 (829.93–830.03) | 42.92 (42.92–42.92) | 8.469 (8.469–8.469) |
| 8192 | 4096 | 20, public fixture | 256 | 3 | 50.13 (50.11–50.19) | 53.51 (53.50–53.51) | 0.434 (0.433–0.434) |
| 8192 | 4096 | 2185, public fixture | 256 | 3 | 722.72 (582.42–723.77) | 47.40 (47.40–47.43) | 3.085 (3.081–3.813) |
| 262152 | 512 | 512, five tasks | 640 | 3 | 320.56 (318.13–320.72) | 43.51 (42.96–43.51) | 1.631 (1.631–1.644) |
| 262152 | 512 | 7000, five tasks | 640 | 3 | 287.80 (285.05–287.80) | 42.38 (41.87–42.40) | 24.357 (24.357–24.593) |
| 262152 | 512 | 20, public fixture | 256 | 3 | 49.02 (48.83–49.14) | 51.53 (50.52–51.53) | 0.444 (0.443–0.446) |
| 262152 | 512 | 2185, public fixture | 256 | 3 | 302.43 (300.92–302.91) | 48.95 (48.52–48.95) | 7.285 (7.281–7.331) |

Values are median (range). Loading/startup is timed separately and excluded
from request timings; the OS weight-file cache stays warm. Fixed expert
placement, **zero prompt reuse**, no adaptation. Prompt/decode rates use the
engine's separate token counts and durations. TTFT measures request start to
first native emitted token. Raw native timing lines, counts, draft acceptance,
input/output hashes and elapsed times: [results.json](results.json),
[CSV](results.csv), [summary](summary.json), [lifecycle](lifecycle.json).
Generated content is discarded; only numeric/hash receipts persist.

Prefill 4096 cuts the 7K TTFT median by 61.9%, but all five long continuations differ from batch 512 and the largest measured 7K decode-cell regression is 16.2%. It is an optional prefill-latency tradeoff, not an overall decode improvement. We keep batch 512 for normal serving. With batch 512, the larger memory profile's 7K TTFT is 24.36s versus 22.23s; 10/10 matched main benchmark cells have identical output hashes across the two memory profiles. See raw per-workload data before generalizing the aggregate.

## Validation and limits

Both 8K profiles pass **21 native reference checks** and **24 API/lifecycle
checks**, including exact target-only/one-draft/MTP4 agreement on the tested
256-output workloads, tools, queued isolation, near-8K recall, 640 outputs,
cancellation during long prefill and restart. Same-profile benchmark repeats
match exactly and all successful native processes exit 0 after graceful QUIT.
The 262K profile separately passes five native checks (target-only/MTP4 at
32K and full, one draft at 32K), with exact 128-token agreement, and 12 API
checks including progressive/full recall, a repeated full request and overflow
rejection. Full-window timing is capacity evidence, separate from the three-run
matched-input tables. These are bounded runtime checks, not broad quality or
roleplay qualification.
[Full-window receipts](full-window-checks.json) keep native and API/SSE timings
separate.

The full registered kernel suite is **26 pass / 4 fail**. Available actual IQ
fixtures and native GGUF checks over eight layers pass. The remaining
registration/fixture, unsupported QFUSE and grouped-S2 bitwise failures are
explained in [QUALIFICATION.md](QUALIFICATION.md), with numeric/reference
receipts in [qualification.json](qualification.json),
[native-reference.json](native-reference.json) and [api-checks.json](api-checks.json).
No claim that the whole kernel suite passes.

Independent GPU, PCIe, memory and thermal guards remain active. Successful runs
have no resets, PCIe errors, swap-out or OOM; peaks are recorded in qualification.json.
Failed qualification attempts are excluded from throughput. Both previously
local mirror safety fixes are upstream: incomplete coverage refuses startup,
and pinned mirror memory is freed before QUIT exit.

The qualified installed profile uses 262144 usable API tokens, prefill 512 and a 4096 MiB reserve. Routed pre/post-reboot full-window, tool, authorization and queued-isolation checks, graceful unload/restart, retained-model recovery and actual Pi chat/read-tool checks pass. Fresh full-window prefill remains about 15–16 minutes; capacity does not imply low latency or broad long-context quality. This report is pinned to v0.1.40.2; v0.1.40.3 was released during testing and is not measured here.

[BUILD.md](BUILD.md) gives exact preparation and portable reproduction steps.
[Portable fixture audit](portable-fixture-audit.json) verifies that the published
drivers produce the measured main inputs and public token fixtures, without
sending another inference request.
