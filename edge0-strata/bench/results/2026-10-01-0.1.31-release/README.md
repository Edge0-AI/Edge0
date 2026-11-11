# Strata 0.1.31 vs 0.1.30 — RX 7900 XTX (gfx1100), 2026-10-01

Two runs of `tools/hip/bench_prefill.py` against an otherwise idle local server, one per release binary.
Sequential same-config runs roughly an hour apart, **not interleaved** — treat the numbers as indicative of
the throughput difference, not as a tightly controlled A/B.

- Host: Ryzen 7 7700X, RX 7900 XTX 24 GB, ROCm 7.1, gfx1100, model on a rotational drive.
- Pack: `coder-iq1_m` (Qwen3.8-Flash-Next-Coder) — identical for both arms, as are template and settings.
- Candidate binary: the **stock v0.1.31** release build, 17,652,664 B,
  sha256 `17f60a01ff553593af813b2ded8b288179ee9285d8a270a10b90f6bb63e5de1d`.
- Source commit for the candidate: `9259cad4cfa3543cd3b8decab5962672b968c649` (tag `v0.1.31`).
- Throughput cap 128 tokens; this is a throughput measurement, **not** a coding-quality benchmark.

## Fresh-prefill trials (fresh > 512 tokens)

| arm | n | prefill tps (min / median / max) | decode tps (min – max) |
|---|---|---|---|
| 0.1.30 (control) | 4 | 225 / 545 / 700 | 16.0 – 34.2 |
| 0.1.31 (candidate) | 4 | 260 / 900 / 994 | 16.7 – 64.7 |

Consistent with the release notes: the warm prefill path and warm decode both improved, with the spread
between trials dominated by first-touch reads from the rotational drive rather than by the engine.

## Files

- `control-0.1.30.json`, `optimized-0.1.31.json` — verbatim `bench_prefill.py` output (all trials,
  including warmup and follow-up, with usage and message text).
- `docs/benchmarks/2026-10-01-gfx1100-0.1.31.json` — the curated control/candidate entry in the
  `docs/benchmarks/` shape, derived from the two files above by `strata-bench-contribute.py`.
