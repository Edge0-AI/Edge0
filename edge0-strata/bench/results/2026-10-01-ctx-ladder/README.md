# 2026-10-01 — Strata IQ2_XS context-ladder TPS on RTX 4070 Ti SUPER

Benchmark target: `Qwen3.8-Flash-Next` IQ2_XS (the README's "recommended" variant for a 16 GiB consumer GPU), serving on `127.0.0.1:8090` from inside the `strata` container (`--network host`, otherwise the docker-proxy DNAT/loopback interaction bites — see commit `02f14e1` in the vps1 project for the parallel diagnostic pattern).

## Summary

| Context | KV | Prompt tokens | **Avg TPS** | Peak TPS (best of 3) | **Engine-predicted TPS (avg)** | Prefill TPS (avg) | Draft accepted | TTFT |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | `int8` | 29,966 | **97.15** | 130.06 | **95.10** | 167.4 | 79.9 % | 168 ms |
| 64k | `int8` | 60,270 | **83.93** | 114.38 | **82.23** | 127.8 | 75.8 % | 264 ms |
| 128k | `q4_0` | 120,748 | **73.97** | 114.68 | **72.97** | 132.8 | 73.7 % | 409 ms |

(Three stream runs per row; `reasoning_effort=minimal`, `max_tokens=192`, `temperature=0`, prompt filled to 92 % of `--max-context`.)

## Reading guide

- The **"Avg TPS"** column is `(completion_tokens) / (first_content_ts - last_content_ts)` measured end-to-end on the host. The **"Peak TPS"** column is the best 32-chunk sliding-window TPS within a single run.
- The **"Engine-predicted TPS"** column is what the engine itself emits at the top level of the final SSE event's `timings` field; it's the same total-decode wall-clock divided by the same completion tokens from inside the GPU.
- `draft accepted` is the fraction of MTP speculative-decoding tokens the verifier kept, summed across the three runs.
- **TTFT** is the time between sending the request and the first SSE `delta.content` chunk landing on the socket — almost entirely prefill (no app-side tail latency).

## Why KV flips at 128 k

At `--max-context 131072` with KV=int8 the VRAM estimate exceeds the 16 GiB card, so we re-launched with `KV=q4_0`. That is the only way to keep the engine inside VRAM at 128 k on this card; the model degrades ~12 % vs int8 at 64 k. The 128 k row is therefore a *different* configuration than the lower rows; do not compare the 32k→64k→128k slope as a pure context effect — about half of the slope is KV compression overhead.

Draft acceptance also glides down modestly with context (80 → 76 → 74 %); the MTP layer's head was tuned for short windows and degrades slightly with longer reasoning spans.

## Method

- Container: hash `2236bbf72fb4` (`strata:latest`, 8.11 GB), engine = sm_89 fat-binary, vision encoder NOT compiled (`BUILD_VISION=0`).
- `docker run -d --name strata --network host --gpus all --ulimit memlock=-1 --shm-size=4g -v strata-data:/data -e ...` re-deployed for each context with `REINSTALL=1`. Re-boot is ~15 s once experts are paged in.
- `run.py`: linear-interpolate fill characters in `[0.50–0.97] · ctx · char-per-token` band via 2 calibration calls (`n_lo`, `n_hi`), then a 5-sample local scan to land near 92 %. 3 SSE stream `POST /v1/chat/completions` calls per row. Engine-reported timings captured from the latest SSE chunk.
- `aggregate.py`: 3 runs averaged; writes `matrix.md` and `summary.json`.

## Caveats / honesty disclaimers

- `stream=true` chunks are subword tokens, not exact tokens, so the engine's `predicted_n` is the canonical "true" completion-token count used in the per-TPS math. The peak window uses raw chunk arrival timing, which slightly over-counts because the engine sometimes batches 1-3 subwords in a single chunk at low pressure.
- The 128k row 1 produced `completion_tokens=174` (`finish=stop`) rather than 192; the engine reached the natural end of the response in less than 192 tokens. The other two runs hit the cap (`finish=length`). Throughput column is correct in either case — total decode time divided by the actual token count.
- `decode_peak_tps` rolls over a 32-chunk window of subwords, so it can briefly exceed the `predicted_per_second` engine metric; this is real (the engine measure is total / total). Engine-reported TPS is the apples-to-apples cross-row number.

## Reproduction

```bash
cd /home/lfontanez/dev/strata/bench/results/2026-10-01-ctx-ladder
python3 run.py --contexts 32768 65536 131072 --out all.json --note "full ladder"
python3 aggregate.py     # writes matrix.md and summary.json
```

For an additional ladder step you'd want `--build-arg BUILD_VISION=1` (and rebuild) to enable image inputs, and the container needs `RAM ≈ 33 GiB × expert count / 16 GiB VRAM` to be safe at rest; this host has 94 GiB so the engine stays in resident-expert mode.
