# 2026-10-10 — community 2× Tesla P40, Qwen3.8-Flash-Next IQ3_S, 131K context

Two Tesla P40 24 GB (sm_61/Pascal), layer split. First IQ3_S datapoint on this pair at 128K
that we know of; extends #1028 (IQ2_XS on the same pair) and the 2×P100 IQ3_S report.

## Machine
- 2× Tesla P40 24 GB (PCIe 3.0 x16), driver 550.144.03
- Xeon E5-2699 v3 (18c/36t, AVX2, no AVX-512), 125 GB RAM, Ubuntu 22.04, SATA-class SSD
- Single machine over; nothing else running.

## Engine / config
- Strata engine **0.1.41** (repo fb58e0d), CUDA 12 source build (`setup.sh --cuda 12`, sm_61
  experimental path), Linux.
- `--expert-cache auto --prefill auto --spec 4 --mtp --max-context 131072 --kv int8
  --kv-resident 32768 --vision --vram-reserve-mib 700 --pcie-frac 0.34 --spec-min-p 0.70`,
  layer split auto → **K=24** (layers 0-23 / 24-47+head), 17 expert-pool workers (default),
  46.84 GiB expert arena in RAM, **9,820 / 24,576 experts resident (17.69 GiB)**, 538 MiB VRAM
  free. Tuned with `setup.sh --calibrate` (kept pcie-frac + spec-min-p only).
- Warm load 45-85 s.

## Prompt / decode (engine timings; temp 0, reasoning_effort none; medians of 3, all values)

| leg (fresh tokens) | prompt tok/s | decode tok/s | TTFT |
|---|---:|---:|---:|
| short system prompt (15-27 tok) | — | **35.0** (34.8/35.0/35.1) | 0.29-0.68 s |
| 3,918 tok | 350.1 | — | 11.2 s |
| 21,858 tok | 558.3 | — | 39.2 s |

- Pre-calibration run (default pcie-frac/spec-min-p): decode 34.2 (31.8/34.2/35.2), prompt
  354.7 / 562.3 — calibration bought +2% decode and draft acceptance 51% → **69%**
  (113/164, 115/187, 112/171 on the decode legs).
- Client-side stream rate matches engine within 0.2 tok/s.
- 128K needle prefill reads: 172-192 s wall including generation (≈680 tok/s effective at
  126.6K fresh tokens).

## Needle recall (tools/needle_bench.py)
6/6 correct — lengths 32k and 128k, depths 10/50/90.

## Conversation checkpoints (pinned golden prefix)
`--conversation-cache-mib 2048 --conversation-cache-slots 4`: a 5.2K-token omp-style system
prompt reads cold in 15.7 s (331 tok/s); every later conversation mounts it in **0.43-0.83 s
wall (cache_n 5,135-5,153)**, stable across churn. A 21.9K prompt re-mounts in 0.30 s TTFT.
`slot save/restore` is refused on the layer split ("session files do not support
--layer-split") — restart persistence is one warm request after READY.

## llama.cpp on the same machine (context)
Qwen3.8-27B-AP-Q4_K_XL, `-sm tensor` both P40s: decode **22.7 tok/s** (llama.cpp v0.4.0-dev
and v0.5.0 agree exactly), prompt ≈388/390 tok/s at the same 3.9K/21.9K fresh-token counts.
`--spec-type draft-mtp --spec-draft-n-max 4` (v0.5.0): 24.9 tok/s (+9.8%, acceptance 33%).
So Flash-Next IQ3_S is +54% decode vs the best 27B config we could build on this pair, ~10%
slower prefill at 4K, +43% prefill at 22K, with 131K vs 393K context. (Flash-Next itself
through stock llama.cpp collapses on prefill — 13-18 tok/s — per sarge18/p40-llm-engine-bakeoff
on identical GPUs.)

## Not tested
Batching (`--parallel`), CPU vision, context >131K, quality suites, AMD, adaptive-swap arms
beyond calibrate's (kept default cadence; 80/160-swap arms measured within noise on its bench).

## Method
3 reps per number, all values + medians published; engine `prompt_per_second` /
`predicted_per_second` from the response `timings`, client-side stream rate as cross-check;
prefill legs are fresh reads (cache_n=0), decode legs greedy; needles via tools/needle_bench.py.
Raw engine-timing lines in `raw-results.md`.
