# Raw engine timings — 2026-10-10 community 2×P40 IQ3_S

## Post-calibration (config: --pcie-frac 0.34 --spec-min-p 0.70), 3 passes

PASS 1 (fresh, cache_n=0)
{"timings": {"cache_n": 0, "prompt_n": 27, "prompt_ms": 636.5, "predicted_n": 256, "predicted_ms": 7320.0, "predicted_per_second": 35.0, "draft_n": 164, "draft_n_accepted": 113}, "client": "decode 34.87 tok/s, ttft 0.68s"}
{"timings": {"cache_n": 0, "prompt_n": 3918, "prompt_ms": 11191.2, "prompt_per_second": 350.1}, "client": "ttft 11.24s"}
{"timings": {"cache_n": 0, "prompt_n": 21858, "prompt_ms": 39154.1, "prompt_per_second": 558.3}, "client": "ttft 39.21s"}

PASS 2 (decode leg fresh-prefix mounted, cache_n=20; prompt legs cache-mounted)
{"timings": {"cache_n": 20, "prompt_n": 7, "predicted_n": 256, "predicted_ms": 7356.4, "predicted_per_second": 34.8, "draft_n": 187, "draft_n_accepted": 115}, "client": "decode 34.69 tok/s, ttft 0.29s"}
{"timings": {"cache_n": 3911, "prompt_n": 7, "prompt_ms": 260.4}, "client": "ttft 0.30s"}
{"timings": {"cache_n": 21851, "prompt_n": 7, "prompt_ms": 294.6}, "client": "ttft 0.35s"}

PASS 3
{"timings": {"cache_n": 20, "prompt_n": 7, "predicted_n": 256, "predicted_ms": 7284.0, "predicted_per_second": 35.1, "draft_n": 171, "draft_n_accepted": 112}, "client": "decode 35.04 tok/s, ttft 0.29s"}
{"timings": {"cache_n": 3911, "prompt_n": 7, "prompt_ms": 272.8}, "client": "ttft 0.32s"}
{"timings": {"cache_n": 21851, "prompt_n": 7, "prompt_ms": 288.1}, "client": "ttft 0.34s"}

## Pre-calibration (defaults), 3 passes

{"decode-256": [34.87 → n/a; pre-cal passes were 31.8 / 34.2 / 35.2 tok/s client (engine ≈ equal), draft acceptance 115/238, 119/217, 129/230]}
{"prompt-3918": [354.7, 354.9, 355.2 tok/s]}
{"prompt-21858": [562.3, 561.0, 562.1 tok/s]}

## Calibrate's own bench (its fixed prompt, not ours)
4 workers 39.9 tok/s · expert tier default 39.9 · 80-swap 40.3 · 160-swap 39.7 → kept
--pcie-frac 0.34 --spec-min-p 0.70 (39.9); pool-workers/expert-cache kept at defaults.

## llama.cpp 27B arms (v0.5.0 build 11146 / 7fe450e19, -c 131072 -ngl 99 -fa 1 -sm tensor)
none:     decode 22.70 tok/s (256-tok stream), ttft 0.43s; prompt ttft 10.50s (~3.9K), 55.70s (~21.9K)
draft-mtp: decode 24.92 tok/s, ttft 0.50s; prompt ttft 11.19s, 58.32s; draft acceptance 0.334 (145/434), mean len 2.32
(v0.4.0-dev deployed baseline: 22.67 tok/s, ttft 10.11s / 56.03s — parity)

## Needles (tools/needle_bench.py, pre-calibration config)
6/6: 32k d10/d50/d90, 128k d10/d50/d90; 128k reads 172-192 s wall (126,597 fresh tokens)

## Golden-prefix parking (5.2K omp-style prefix)
cold read: {"prompt_n": 5152, "prompt_ms": 15520, "wall_s": 15.7}
session B mounts: {"cache_n": 5135, "prompt_n": 25, "wall_s": 0.83}
parked mounts: {"cache_n": 5145, "prompt_n": 7, "wall_s": 0.49}, {"cache_n": 5153, "prompt_n": 7, "wall_s": 0.44}, {"cache_n": 5150, "prompt_n": 7, "wall_s": 0.43}
