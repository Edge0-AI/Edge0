# Strata TPS benchmark results - Shin-BlackMamba - 2026-10-05 (corrected)

Engine: Strata Qwen3.8-Flash-Next on RTX 4070 Ti SUPER (16 GiB, sm_89).
Image: strata:latest, sm_89 fat-binary, vision decoder NOT compiled.
Container: --network host --gpus all --ulimit memlock=-1 --shm-size=4g.
Bench harness: `run.py` issues, per row:
  1. **Cold prefill probe** (max_tokens=1, non-stream): `timings.prompt_per_second` after `POST /unload`
  2. **Three SSE stream calls** with reasoning_effort=minimal, max_tokens=192, t=0

KV choices per row:
  - IQ2_XS rows use KV=int8. **CORRECTED 2026-10-05** - KV-streaming keeps KV table small in RAM so int8 is viable for ALL IQ2_XS contexts.
  - IQ3_XXS rows at 32k/64k use KV=int8; at 128k/256k use KV=q4_0/k8v4 because IQ3_XXS residents 47 GB of experts and KV=int8 would push past VRAM at > 64 k.

Two prefill columns are reported:
  - **Cold Prefill** = true end-of-prompt throughput (after /unload / re-fetch).
  - **SSE Prefill**  = engine-reported time-to-first-chunk throughput during streaming, lower because --prefill auto overlaps decode with the next 8K-token chunk's prefill.

## Matrix

| Label | Model | KV | Prompt tokens | Avg TPS | Peak TPS (best of 3) | Engine TPS (avg) | Cold Prefill | SSE Prefill | Draft accepted | TTFT (avg) |
|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32k | IQ2_XS | int8 | 29,966 | **95.62** | 136.34 | **93.47** | **2694.8** | 141.3 | 75.8% | 183 ms |
| 64k | IQ2_XS | int8 | 60,270 | **91.80** | 123.20 | **89.90** | **2678.0** | 140.9 | 74.5% | 271 ms |
| 128k | IQ2_XS | int8 | 120,748 | **87.95** | 117.52 | **86.67** | **2670.4** | 132.6 | 76.8% | 421 ms |
| 256k | IQ2_XS | int8 | 240,912 | **79.63** | 107.36 | **78.50** | **2483.9** | 127.3 | 69.6% | 725 ms |
| IQ3_XXS @32k | IQ3_XXS | int8 | 29,966 | **73.32** | 97.96 | **71.73** | **2747.4** | 116.5 | 69.8% | 216 ms |
| IQ3_XXS @64k | IQ3_XXS | int8 | 60,270 | **69.16** | 98.66 | **67.70** | **2544.8** | 107.0 | 70.3% | 295 ms |
| IQ3_XXS @128k | IQ3_XXS | q4_0 | 120,748 | **65.51** | 84.42 | **64.10** | **2477.9** | 105.2 | 67.7% | 461 ms |
| IQ3_XXS @256k | IQ3_XXS | k8v4 | 240,912 | **53.70** | 71.09 | **52.97** | **2403.9** | 91.9 | 66.6% | 768 ms |

## Quality-vs-Speed comparison (IQ2_XS vs IQ3_XXS at same KV)

| Context | KV | IQ2_XS engine TPS | IQ3_XXS engine TPS | IQ3 decode-TPS tax | IQ3 peak tax |
|---|---|---:|---:|---:|---:|
| 32k | int8 | 93.47 | 71.73 | -23.3% | -28.2% |
| 64k | int8 | 89.90 | 67.70 | -24.7% | -19.9% |
| 128k | int8 | 86.67 | 64.10 | -26.0% | -28.2% |
| 256k | int8 | 78.50 | 52.97 | -32.5% | -33.8% |

## Per-row audit trails

- [32k](./logs/32k.md)
- [64k](./logs/64k.md)
- [128k](./logs/128k.md)
- [256k](./logs/256k.md)
- [IQ3_XXS @32k](./logs/IQ3_XXS_at32k.md)
- [IQ3_XXS @64k](./logs/IQ3_XXS_at64k.md)
- [IQ3_XXS @128k](./logs/IQ3_XXS_at128k.md)
- [IQ3_XXS @256k](./logs/IQ3_XXS_at256k.md)

## Per-run detail

### 32k (model=IQ2_XS, KV=int8, prompt=29,966 tok)

  Cold prefill probe: prompt_n=29966 prompt_total_ms=11119.9 prompt_per_second=2694.8 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 29,966 | 192 | 189 ms | 85.69 | 108.05 | 83.70 | 127.3 | 120/161 | length |
| 2 | 29,966 | 192 | 161 ms | 103.62 | 136.34 | 101.60 | 159.8 | 127/164 | length |
| 3 | 29,966 | 192 | 199 ms | 97.56 | 124.34 | 95.10 | 136.7 | 117/155 | length |

### 64k (model=IQ2_XS, KV=int8, prompt=60,270 tok)

  Cold prefill probe: prompt_n=60270 prompt_total_ms=22505.8 prompt_per_second=2678.0 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 60,270 | 192 | 280 ms | 82.98 | 109.92 | 81.20 | 129.9 | 114/153 | length |
| 2 | 60,270 | 192 | 261 ms | 96.19 | 117.86 | 94.30 | 149.9 | 118/160 | length |
| 3 | 60,270 | 192 | 272 ms | 96.22 | 123.20 | 94.20 | 142.9 | 116/154 | length |

### 128k (model=IQ2_XS, KV=int8, prompt=120,748 tok)

  Cold prefill probe: prompt_n=120748 prompt_total_ms=45217.9 prompt_per_second=2670.4 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 120,748 | 192 | 432 ms | 82.28 | 111.19 | 81.10 | 123.0 | 115/151 | length |
| 2 | 120,748 | 192 | 403 ms | 89.98 | 117.52 | 88.70 | 143.4 | 116/147 | length |
| 3 | 120,748 | 192 | 430 ms | 91.58 | 116.04 | 90.20 | 131.3 | 117/155 | length |

### 256k (model=IQ2_XS, KV=int8, prompt=240,912 tok)

  Cold prefill probe: prompt_n=240912 prompt_total_ms=96989.1 prompt_per_second=2483.9 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 240,912 | 192 | 721 ms | 73.64 | 107.12 | 72.60 | 119.5 | 112/171 | length |
| 2 | 240,912 | 192 | 741 ms | 81.55 | 107.36 | 80.50 | 138.6 | 112/157 | length |
| 3 | 240,912 | 192 | 714 ms | 83.69 | 106.44 | 82.40 | 123.7 | 111/153 | length |

### IQ3_XXS @32k (model=IQ3_XXS, KV=int8, prompt=29,966 tok)

  Cold prefill probe: prompt_n=29966 prompt_total_ms=10907.2 prompt_per_second=2747.4 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 29,966 | 192 | 219 ms | 66.09 | 80.30 | 64.60 | 107.4 | 117/172 | length |
| 2 | 29,966 | 192 | 209 ms | 76.03 | 97.96 | 74.60 | 133.8 | 119/167 | length |
| 3 | 29,966 | 192 | 222 ms | 77.85 | 95.92 | 76.00 | 108.4 | 120/171 | length |

### IQ3_XXS @64k (model=IQ3_XXS, KV=int8, prompt=60,270 tok)

  Cold prefill probe: prompt_n=60270 prompt_total_ms=23683.4 prompt_per_second=2544.8 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 60,270 | 192 | 328 ms | 62.02 | 79.17 | 60.70 | 95.4 | 120/174 | length |
| 2 | 60,270 | 192 | 272 ms | 72.48 | 90.30 | 71.10 | 117.4 | 121/172 | length |
| 3 | 60,270 | 192 | 285 ms | 72.98 | 98.66 | 71.30 | 108.2 | 118/165 | length |

### IQ3_XXS @128k (model=IQ3_XXS, KV=q4_0, prompt=120,748 tok)

  Cold prefill probe: prompt_n=120748 prompt_total_ms=48729.8 prompt_per_second=2477.9 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 120,748 | 192 | 491 ms | 57.42 | 73.81 | 56.20 | 97.9 | 110/173 | length |
| 2 | 120,748 | 192 | 430 ms | 67.33 | 84.42 | 66.10 | 115.1 | 115/170 | length |
| 3 | 120,748 | 192 | 463 ms | 71.77 | 83.03 | 70.00 | 102.6 | 118/164 | length |

### IQ3_XXS @256k (model=IQ3_XXS, KV=k8v4, prompt=240,912 tok)

  Cold prefill probe: prompt_n=240912 prompt_total_ms=100219.1 prompt_per_second=2403.9 t/s

| Run | Prompt tok | Completion | TTFT | Avg TPS | Peak TPS | Engine TPS | SSE Prefill (per-run rpt) | Draft | Finish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 240,912 | 192 | 764 ms | 49.46 | 69.46 | 48.80 | 89.9 | 112/173 | length |
| 2 | 240,912 | 192 | 762 ms | 54.79 | 67.62 | 54.10 | 96.4 | 109/163 | length |
| 3 | 240,912 | 192 | 779 ms | 56.85 | 71.09 | 56.00 | 89.4 | 114/167 | length |
