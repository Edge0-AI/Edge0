# Local cache measurements

Measured September 12, 2026 on an Apple M5 MacBook Air with 24 GB unified memory,
local `edge0-8b`, its shipped LoRA and prerouter, MLX 0.30.6, and mlx-lm 0.31.0.
The fixed coding fixture contains 2,799 prompt tokens. Each request generates eight
greedy tokens. The appended turn and branch contain 2,832 and 2,831 prompt tokens.
The interval is 2,048; the checkpoint budget is 20 GiB.

These are sequential single samples on an active laptop, not confidence intervals.
The OS page cache was warm. The baseline runs first and therefore includes more
kernel/expert warmup. Initial model loading and artifact hashing are excluded from
request latency. A fresh process is measured separately; no claim of cold-SSD
performance is made. [Raw measurements](benchmarks/conversation-cache-m5.json)
include output token IDs, memory, decode, and disk counters.

| Request | Reused tokens | TTFT (s) | Wall (s) | Restore (s) | Remaining prefill (s) | Writes (s) |
|---|---:|---:|---:|---:|---:|---:|
| Disabled | 0 | 6.514 | 7.465 | — | 6.319 | — |
| First write | 0 | 6.775 | 7.765 | — | 5.296 | 1.937 |
| Repeat | 2,799 | 0.503 | 1.022 | 0.388 | <0.001 | 0 |
| Appended turn | 2,807 | 1.619 | 2.568 | 0.320 | 0.580 | 1.207 |
| Branch | 2,807 | 1.532 | 2.667 | 0.328 | 0.468 | 1.226 |
| Process restart | 2,799 | 0.610 | 1.224 | 0.395 | <0.001 | 0 |

Write time includes prompt and completed-generation snapshots, so not all of it
falls before the first token. Prompt and completion counts remain unchanged by
reuse. Disabled, first-write, repeat, and restarted requests produced identical
eight-token outputs. Appended/branched requests processed only 25/24 unmatched
tokens through prefill.

At this measured prefix length, the first cached request cost 0.261 s extra TTFT
and 0.301 s extra total time. One repeat saved 6.010 s TTFT and 6.442 s total time
against the disabled sample, so the first repeat amortized the observed initial
cost. Even charging the full 1.937 s publication time as overhead, one repeat
covered it. This is a measured **reuse-count** break-even at 2,799 tokens, not a
measured minimum token-length threshold. Short-prompt break-even, randomized
request-order trials, sustained workloads, and cold filesystem trials remain
unmeasured. Earlier development samples ranged from 5.45–6.36 s disabled TTFT
and 0.49 s repeat TTFT; laptop load and warmup affect the absolute values.

| Request | Peak MLX (GiB) | Sampled peak RSS (GiB) | Decode (tokens/s) |
|---|---:|---:|---:|
| Disabled | 2.54 | 4.57 | 7.10 |
| First write | 3.16 | 4.73 | 15.47 |
| Repeat | 1.66 | 4.64 | 12.73 |
| Appended turn | 1.69 | 4.65 | 17.71 |
| Branch | 1.71 | 4.85 | 12.55 |
| Process restart | 1.11 | 1.97 | 9.82 |

Publication increased peak MLX allocation by about 24% in this sample. The active
attention cache still lives in RAM; caching is not active-context offloading.
RSS includes expert caches and other allocations, and is sampled every 20 ms.
Decode values cover only eight tokens and have substantial warmup/noise; they do
not establish a decode-speed improvement. Including final synchronous publication,
the effective rate after the first token was 7.07 tokens/s for first write versus
13.49 for repeat.

Unique payloads occupied 761,533,603 bytes after the first conversation, then
1,223,715,239 bytes after both branches. System-wide disk counters reported about
1.42 GB written during first write, 0.78 GB for the appended turn, and 0.91 GB for
the branch. These include other processes and filesystem behavior. Per-process
block counters remained zero on this macOS run. Content deduplication saves
retained storage, but the current synchronous codec reserializes shared blocks;
it does not eliminate their write/checksum work.

A separate radix-only benchmark performed 10,000 lookups on 130-token keys:

| Checkpoints | Mean lookup (µs) |
|---:|---:|
| 1,000 | 1.55 |
| 10,000 | 1.57 |
| 100,000 | 1.56 |

This isolates index traversal. It excludes SQLite startup/rebuild, lock contention,
and payload restore. End-to-end warm lookup was about 0.27–0.35 ms in the verified
run; first lookup after restart, including index construction, was 3.61 ms.

## Correctness coverage and remaining limits

The tests cover radix branches and exact hits, incremental local index updates,
namespace/artifact invalidation, reference cleanup and LRU eviction, physical
storage limits, missing/corrupt payloads and metadata, repair of shared corrupt
blocks, interrupted publication including abrupt process exit, and concurrent
threads/processes. Continuation coverage includes attention plus recurrent state,
family prerouter fields and expert staging, exact hits, one-token suffixes,
intermediate prefill boundaries, full sampling history, EOS, generation limits,
and callback cancellation. HTTP session tests verify released idle context and
unchanged total token accounting.

Real 8B continuation logits matched within `rtol=1e-4, atol=1e-4`; restored prompt
logits used `1e-5`. The actual small Qwen gated-delta/attention backbone also passed
with random weights at `1e-5`. Real-weight **35B validation remains outstanding**
because that checkpoint is unavailable locally.

A checkpoint resumes the saved execution boundary. Hybrid prerouter behavior can
depend on prefill versus decode execution and chunk boundaries; the continuation
contract is equality with the same uninterrupted saved trajectory, not arbitrary
re-chunking of a previously decoded conversation. Corrupt entries become misses;
whole-database destruction or cache filesystem failure is not a recovery mechanism
for the inference service itself. No background write queue or quantized KV format
is included in this phase.
