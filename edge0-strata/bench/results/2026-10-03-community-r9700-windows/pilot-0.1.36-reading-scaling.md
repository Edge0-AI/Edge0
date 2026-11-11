# Strata R9700 / Windows 11 — file-read scaling (measured, from strata-iq2_xs.log deltas)

Counter: `strata serve: expert tiers: ... since the start RAM <n> blobs, files <n> blobs <MB> MB read (the GGUF in place)`
Deltas between consecutive requests inside the SAME engine session.

| prompt tokens | new tokens read | file MB read this request | MB per new token |
|---|---|---|---|
| 406 | 406 | +581 | 1.43 |
| 1,042 | 296 | +727 | 2.46 |
| 1,997 | 959 | +1,639 | 1.71 |
| 11,042 | 9,050 | +12,707 | 1.40 |
| 53,470 | 42,433 | +31,666 | 0.75 |
| 125,247 | 71,782 | +45,238 | 0.63 |
| 228,869 | 103,402 | +65,285 | 0.63 |

(earlier small requests: 406 tok → +581 MB, 492 tok/91 new → +581 MB)

## Findings
- File reads scale **with prompt length** (~0.6–1.4 MB per new token) → the PLE/expert data is streamed from the SSD per prompt token (matches docs: "a few rows per token through the OS cache"). This caps prefill at ~1,000–1,300 tok/s.
- NOT a one-time cost per engine start (earlier hypothesis — corrected).
- Genuine one-off: the FIRST request after each engine start (93 tok took 39.0 s vs an identical ~91-tok later request at 4.7 s).
- Per-request fixed overhead ≈ 4.4 s (linear fit a=4.4 s, marginal rate ≈ 1,300 tok/s; a naive fit including the cold-start outlier gives a=12.4 s).

## Cold-start evidence
`strata generate: resident RAM mode: 6.95 GiB of experts in RAM (page-locked), 19403 in the GPU cache`
`FileExpertSource: mapped pinned cache complement ready: resident 6.95 GiB, pinned 6.95 GiB` (identical with --resident-budget-gib 16 and 30)
